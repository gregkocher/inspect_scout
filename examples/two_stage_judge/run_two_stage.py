#!/usr/bin/env python3
"""Run the two-stage judge: triage on every transcript, then review on the escalated ones.

Escalated = triage score >= --threshold, plus a random --audit-fraction of the transcripts
below it, so the review stage also measures how often triage misses a real problem.

    python -m examples.two_stage_judge.run_two_stage -T <transcripts> --scans scans_two_stage --threshold 4

(run from the repository root)

Models default to those in two_stage_scanners.py; override with --triage-model/--review-model.
Writes two_stage_summary.csv into the scans directory and prints a short summary.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
from inspect_ai.model import GenerateConfig
from inspect_scout import scan, scan_results_df, transcripts_from

from .two_stage_scanners import REVIEW_MODEL, TRIAGE_MODEL, review_scan, triage_scan


def results(location: str, scanner: str) -> pd.DataFrame:
    df = scan_results_df(location).scanners[scanner]
    keep = [
        "transcript_id",
        "transcript_task_id",
        "value",
        "label",
        "explanation",
        "scan_error",
    ]
    return df[[c for c in keep if c in df.columns]].copy()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "-T", "--transcripts", required=True, help="transcript database or logs"
    )
    ap.add_argument(
        "--scans", default="scans_two_stage", help="where scan results are written"
    )
    ap.add_argument(
        "--threshold", type=float, default=4, help="triage score that escalates"
    )
    ap.add_argument(
        "--audit-fraction",
        type=float,
        default=0.1,
        help="share of non-escalated to review",
    )
    ap.add_argument("--triage-model", default=TRIAGE_MODEL)
    ap.add_argument("--review-model", default=REVIEW_MODEL)
    ap.add_argument("--max-connections", type=int, default=50)
    ap.add_argument(
        "--limit", type=int, default=None, help="only the first N transcripts"
    )
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    config = GenerateConfig(max_connections=a.max_connections)
    transcripts = transcripts_from(a.transcripts)
    if a.limit:
        transcripts = transcripts.limit(a.limit)

    print(f"== stage 1: triage with {a.triage_model}")
    s1 = scan(
        scanners=[triage_scan(model=a.triage_model)],
        transcripts=transcripts,
        scans=a.scans,
        model_config=config,
        display="none",
    )
    tri = results(s1.location, "triage_scan")
    errors = tri["scan_error"].notna().sum() if "scan_error" in tri else 0

    flagged = tri[tri["value"] >= a.threshold]["transcript_id"].tolist()
    rest = tri[tri["value"] < a.threshold]["transcript_id"].tolist()
    rng = random.Random(a.seed)
    audit = rng.sample(rest, k=round(len(rest) * a.audit_fraction)) if rest else []
    to_review = flagged + audit
    print(
        f"   {len(tri)} triaged ({errors} errors), {len(flagged)} escalated (score >= {a.threshold}), "
        f"{len(audit)} audit sample"
    )

    rev = pd.DataFrame(columns=["transcript_id", "value", "label", "explanation"])
    if to_review:
        print(f"== stage 2: review with {a.review_model}")
        ids = ",".join(f"'{t}'" for t in to_review)
        s2 = scan(
            scanners=[review_scan(model=a.review_model)],
            transcripts=transcripts_from(a.transcripts).where(
                f"transcript_id IN ({ids})"
            ),
            scans=a.scans,
            model_config=config,
            display="none",
        )
        rev = results(s2.location, "review_scan")

    summary = tri.rename(
        columns={
            "value": "triage_score",
            "label": "triage_category",
            "explanation": "triage_explanation",
        }
    ).merge(
        rev[["transcript_id", "value", "label", "explanation"]].rename(
            columns={
                "value": "review_score",
                "label": "review_category",
                "explanation": "review_explanation",
            }
        ),
        on="transcript_id",
        how="left",
    )
    summary["stage2_reason"] = summary["transcript_id"].map(
        lambda t: "escalated" if t in flagged else "audit" if t in audit else ""
    )
    summary = summary.sort_values(["review_score", "triage_score"], ascending=False)
    out = Path(a.scans) / "two_stage_summary.csv"
    summary.to_csv(out, index=False)

    reviewed = summary[summary["review_score"].notna()]
    confirmed = reviewed[
        (reviewed["stage2_reason"] == "escalated")
        & (reviewed["review_score"] >= a.threshold)
    ]
    missed = reviewed[
        (reviewed["stage2_reason"] == "audit")
        & (reviewed["review_score"] >= a.threshold)
    ]
    print(
        f"\nescalated confirmed by review (review >= {a.threshold}): {len(confirmed)}/{len(flagged)}"
    )
    print(f"audit sample that review flags (triage misses): {len(missed)}/{len(audit)}")
    cols = [
        "transcript_task_id",
        "triage_score",
        "triage_category",
        "review_score",
        "review_category",
        "stage2_reason",
    ]
    print(
        summary[[c for c in cols if c in summary.columns]]
        .head(15)
        .to_string(index=False)
    )
    print(f"\nsummary: {out}\nview:    scout view -T {a.transcripts} --scans {a.scans}")


if __name__ == "__main__":
    main()
