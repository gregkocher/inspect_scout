#!/usr/bin/env python3
"""Convert a folder of OpenAI Chat Completions trajectories into an Inspect Scout transcript database.

Input files (``*.json`` or ``*.jsonl``, searched recursively) may each hold:
  - a list of messages:                         [{"role": "system", ...}, ...]
  - an object with a "messages" key:            {"messages": [...], "tools": [...], "metadata": {...}}
  - (.jsonl) one such object or list per line.

The heavy lifting (tool-call parsing, linking tool results to calls, reasoning extraction) is done by
Inspect's own ``inspect_ai.model.messages_from_openai``. This script only:
  1. finds the files and builds one Scout ``Transcript`` per trajectory, and
  2. normalizes a few real-world deviations from the strict Chat Completions format first
     (each one is counted and reported, nothing is changed silently):
       - keys whose value is null (e.g. parquet exports write ``tool_calls: null`` on every message);
       - ``content: null`` on an assistant turn that has tool calls -> ``""``. Without this, Inspect
         moves a later text-only assistant reply into that earlier turn (a deliberate fix-up for
         Codex-CLI transcripts), which changes message order;
       - tool calls missing ``"type": "function"``;
       - SWE-agent style ``tool_call_ids: [id]`` on tool results -> standard ``tool_call_id``;
       - ``"type": "custom"`` tool calls (Inspect silently drops them) -> function calls.
  3. checks every converted trajectory (message, tool-call and tool-result counts and links) and
     marks the file FAILED instead of writing a silently damaged transcript.

Usage:
  python3 openai_to_scout.py <input_dir> <transcripts_db_dir> [--source-type NAME] [--overwrite]
  scout view -T <transcripts_db_dir>
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import copy
import hashlib
import json
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

from inspect_ai.model import ChatMessage, messages_from_openai
from inspect_scout import Transcript, transcripts_db
from openai.types.chat import ChatCompletionMessageParam

try:  # Scout rejects metadata keys that clash with its own columns (e.g. 'model')
    from inspect_scout._transcript.database.schema import reserved_columns

    RESERVED = set(reserved_columns())
except Exception:
    RESERVED = set(Transcript.model_fields)


# --------------------------------------------------------------------------------------------- reading
def iter_records(path: Path) -> Iterator[tuple[int, Any]]:
    """Yield (record_index, record) for every trajectory in a .json / .jsonl file."""
    if path.suffix == ".jsonl":
        for i, line in enumerate(path.read_text().splitlines()):
            if line.strip():
                yield i, json.loads(line)
    else:
        yield 0, json.loads(path.read_text())


def split_record(
    rec: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None, dict[str, Any]]:
    """Return (messages, tools, metadata) from a record that is a list or a dict."""
    if isinstance(rec, list):
        return rec, None, {}
    if isinstance(rec, dict) and isinstance(rec.get("messages"), list):
        meta = dict(rec.get("metadata") or {})
        for k, v in rec.items():  # keep other small scalar top-level fields as metadata
            if k not in ("messages", "tools", "metadata") and isinstance(
                v, (str, int, float, bool)
            ):
                meta.setdefault(k, v)
        return rec["messages"], rec.get("tools"), meta
    raise ValueError(
        "record is neither a message list nor an object with a 'messages' list"
    )


# --------------------------------------------------------------------------------------- normalizing
def normalize(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], collections.Counter[str]]:
    fixes: collections.Counter[str] = collections.Counter()
    out = []
    for m in copy.deepcopy(messages):
        for k in [k for k, v in m.items() if v is None]:
            if not (k == "content" and m.get("role") == "assistant"):
                del m[k]
                fixes[f"dropped null '{k}'"] += 1
        if m.get("role") == "assistant" and m.get("tool_calls"):
            if m.get("content") is None:
                m["content"] = ""
                fixes["assistant content null->'' (prevents reorder)"] += 1
            for j, tc in enumerate(m["tool_calls"]):
                if (
                    tc.get("type") == "custom"
                ):  # Inspect silently DROPS these, so convert them
                    cu = tc.get("custom") or {}
                    m["tool_calls"][j] = {
                        "id": tc.get("id"),
                        "type": "function",
                        "function": {
                            "name": cu.get("name", "custom"),
                            "arguments": json.dumps({"input": cu.get("input")}),
                        },
                    }
                    fixes[
                        "custom tool call -> function (input kept as {'input': ...})"
                    ] += 1
                elif "type" not in tc:
                    tc["type"] = "function"
                    fixes["added tool_call type=function"] += 1
        elif m.get("role") == "assistant" and m.get("content") is None:
            m["content"] = ""
            fixes["assistant content null->''"] += 1
        if (
            m.get("role") == "tool"
            and "tool_call_id" not in m
            and m.get("tool_call_ids")
        ):
            ids = m.pop("tool_call_ids")
            m["tool_call_id"] = ids[0]
            fixes["tool_call_ids[0]->tool_call_id"] += 1
            if len(ids) > 1:
                fixes["WARNING tool result references >1 call id"] += 1
        out.append(m)
    return out, fixes


# ----------------------------------------------------------------------------------------- checking
def integrity_problems(
    msgs: list[dict[str, Any]], chat: list[ChatMessage]
) -> list[str]:
    """Cheap structural checks that catch silent drops or mis-links in the conversion."""
    probs = []
    if len(msgs) != len(chat):
        probs.append(f"message count {len(msgs)} -> {len(chat)}")
    n_in = sum(
        len(m.get("tool_calls") or []) for m in msgs if m.get("role") == "assistant"
    )
    n_out = sum(
        len(getattr(m, "tool_calls", None) or []) for m in chat if m.role == "assistant"
    )
    if n_in != n_out:
        probs.append(f"tool calls {n_in} -> {n_out}")
    call_ids = {
        tc.id for m in chat if m.role == "assistant" for tc in (m.tool_calls or [])
    }
    for m in chat:
        if m.role == "tool" and (
            m.tool_call_id is None or m.tool_call_id not in call_ids
        ):
            probs.append(f"tool result {m.tool_call_id!r} not linked to a tool call")
    return probs


# ------------------------------------------------------------------------------------------ building
async def build_transcripts(
    input_dir: Path, source_type: str
) -> tuple[list[Transcript], list[dict[str, Any]]]:
    transcripts: list[Transcript] = []
    report: list[dict[str, Any]] = []
    files = sorted([p for p in input_dir.rglob("*") if p.suffix in (".json", ".jsonl")])
    for path in files:
        for idx, rec in iter_records(path):
            rel = path.relative_to(input_dir).as_posix() + (
                f"#{idx}" if path.suffix == ".jsonl" else ""
            )
            try:
                raw_msgs, tools, meta = split_record(rec)
                msgs, fixes = normalize(raw_msgs)
                chat = await messages_from_openai(
                    cast(list[ChatCompletionMessageParam], msgs),
                    model=meta.get("model"),
                )
                problems = integrity_problems(msgs, chat)
                if problems:
                    raise ValueError(
                        "integrity check failed: " + "; ".join(problems[:5])
                    )
            except Exception as e:  # report and continue, never drop silently
                report.append(
                    {
                        "file": rel,
                        "status": "FAILED",
                        "error": f"{type(e).__name__}: {e}"[:300],
                    }
                )
                continue
            tid = hashlib.sha1(f"{source_type}:{rel}".encode()).hexdigest()[:16]
            meta = {
                **meta,
                "source_file": rel,
                "n_input_messages": len(raw_msgs),
                "normalization": dict(fixes),
            }
            model_name = meta.get("model")
            meta = {(f"src_{k}" if k in RESERVED else k): v for k, v in meta.items()}
            if tools:
                meta["tool_names"] = sorted(
                    {(t.get("function") or {}).get("name", "?") for t in tools}
                )
            transcripts.append(
                Transcript(
                    transcript_id=tid,
                    source_type=source_type,
                    source_id=rel,
                    source_uri=str(path.resolve()),
                    model=model_name,
                    task_set=str(
                        meta.get("source") or meta.get("src_source") or input_dir.name
                    ),
                    task_id=str(
                        meta.get("instance_id")
                        or meta.get("src_task_id")
                        or meta.get("task_id")
                        or Path(rel).stem
                    ),
                    success=meta.get("resolved")
                    if isinstance(meta.get("resolved"), bool)
                    else None,
                    message_count=len(chat),
                    messages=chat,
                    metadata=meta,
                )
            )
            report.append(
                {
                    "file": rel,
                    "status": "ok",
                    "transcript_id": tid,
                    "in": len(raw_msgs),
                    "out": len(chat),
                    "fixes": dict(fixes),
                }
            )
    return transcripts, report


async def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("db_dir", type=Path)
    ap.add_argument("--source-type", default="openai_chat_completions")
    ap.add_argument(
        "--overwrite", action="store_true", help="delete db_dir first if it exists"
    )
    a = ap.parse_args()
    if a.db_dir.exists():
        if not a.overwrite:
            sys.exit(f"{a.db_dir} exists; pass --overwrite to replace it")
        shutil.rmtree(a.db_dir)
    transcripts, report = await build_transcripts(a.input_dir, a.source_type)
    async with transcripts_db(str(a.db_dir)) as db:
        await db.insert(transcripts)
    for r in report:
        if r["status"] == "ok":
            flag = "" if r["in"] == r["out"] else "  <-- message count changed"
            print(
                f"ok      {r['file'][:70]:70s} {r['in']:4d} -> {r['out']:4d}{flag}  {r['fixes'] or ''}"
            )
        else:
            print(f"FAILED  {r['file']}: {r['error']}")
    n_ok = sum(r["status"] == "ok" for r in report)
    print(
        f"\n{n_ok}/{len(report)} trajectories written to {a.db_dir}\nView with:  scout view -T {a.db_dir}"
    )


if __name__ == "__main__":
    asyncio.run(main())
