# Importing OpenAI Chat Completions trajectories

[`openai_to_scout.py`](openai_to_scout.py) converts a folder of agent
trajectories in the OpenAI Chat Completions message format into a Scout
transcript database, which you can then view and scan.

```bash
python3 openai_to_scout.py <input_dir> <transcripts_db_dir> [--source-type NAME] [--overwrite]
scout view -T <transcripts_db_dir>
```

## Input

Every `*.json` and `*.jsonl` file under `<input_dir>` (searched recursively).
Each trajectory can be:

- a list of messages: `[{"role": "system", ...}, {"role": "user", ...}, ...]`
- an object with a `messages` list, plus optional `tools` (the tool schemas)
  and `metadata`: `{"messages": [...], "tools": [...], "metadata": {...}}`

A `.jsonl` file holds one trajectory per line. Other scalar top-level fields of
an object (e.g. `"id"`, `"label"`) are kept as transcript metadata.

Other layouts (a list of trajectories in one `.json` file, logged API
request/response pairs, one API call per line) are not read as-is; adapt
`iter_records()` and `split_record()` to them. Such files are reported as
`FAILED`, never skipped silently.

## What it does

The conversion itself is inspect_ai's `messages_from_openai()`. Before it, the
script normalizes deviations from the strict format that real datasets contain
and reports every fix per file:

| Deviation | Fix | Why |
|---|---|---|
| keys whose value is `null` (e.g. `tool_calls: null` from parquet exports) | dropped | |
| `content: null` on an assistant turn with tool calls | `""` | otherwise a later text-only assistant reply is merged into that turn, changing message order |
| tool calls without `"type"` | `"function"` | |
| SWE-agent style `tool_call_ids: [id]` on tool results | `tool_call_id: id` | otherwise the conversion fails |
| `"type": "custom"` tool calls | function calls with arguments `{"input": ...}` | otherwise they are dropped silently |

After converting, it checks that the message count, the tool-call count and
the order of roles are unchanged (counting OpenAI's `developer` role as
`system`, as Inspect does) and that every tool result links to a tool call. A
trajectory that fails these checks is reported as `FAILED` instead of being
written.

It also reports tool calls that no tool result answers, without failing the
trajectory:

- in the **last assistant turn**: usually the run just ended there (e.g. a final
  `finish` or `submit` call); counted in the summary;
- in an **earlier turn**: the log was probably cut or edited, or the result was
  lost; printed as a `WARNING` line for that trajectory.

Both lists are stored in the transcript's metadata as `unanswered_tool_calls`
(`final_turn` and `earlier_turns`, each entry `function#call_id@message_index`).

Each transcript gets:

- a deterministic `transcript_id` (from the source type and file path), so
  re-running produces the same ids;
- `source_type`, `source_id` (relative file path) and `source_uri`;
- `model`, `task_set`, `task_id` and `success` from the trajectory's metadata
  when present (`model`, `source`, `instance_id`/`task_id`, `resolved`);
- metadata with the original fields plus `source_file`, `n_input_messages`,
  `normalization` (the fixes applied), `unanswered_tool_calls` and `tool_names`. Metadata keys that clash
  with Scout's own columns (e.g. `model`) are stored as `src_<key>`.

The script prints one line per trajectory (`ok` with message counts and fixes,
or `FAILED` with the reason) and a total.
