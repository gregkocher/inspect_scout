# Two-stage LLM judging: cheap triage, then careful review

Two whole-transcript LLM scanners meant to be used as a cascade:

| | `triage_scan` | `review_scan` |
|---|---|---|
| Purpose | screen every transcript quickly and cheaply | judge the escalated transcripts carefully |
| Default model (model role) | `openai/gpt-4.1-mini` (`triage`) | `openai/gpt-5.5` (`review`) |
| Prompt | short, tuned for **recall**: missing a problem is worse than a false flag; when unsure, score higher | structured for **calibration**: reasoning steps written before the score, agent-vs-user split, anchored 1-10 rubric, adjustment factors, "why not one higher or lower", optional worked example |
| Score (1-10) | review priority | severity, driven by the most severe real issue |
| Structured answer | candidate issues (category, message ids, note), score, primary category, explanation | analysis, issues (category, message ids, verbatim evidence, severity), score, primary category, confidence, explanation |

Both share one set of issue categories (instruction violation, deception,
destructive action, secret exposure, PII exposure, unsafe execution, reward
hacking, scope creep, environment issue, other), show the system prompt to the
judge, and use a nonce so text in the transcript cannot pass for message
markers. Both prompts are first drafts; refine them after reading the data and
the judges' outputs. `WORKED_EXAMPLE` in `two_stage_scanners.py` is an empty
slot for one fully graded example.

## Running one stage

```bash
scout scan examples/two_stage_judge/two_stage_scanners.py@triage_scan -T <transcripts> \
    --max-connections 100
scout scan examples/two_stage_judge/two_stage_scanners.py@review_scan -T <transcripts> \
    -F "transcript_id IN ('id1', 'id2')" --model-role review=openai/gpt-5.5
```

`--model-role triage=...` / `--model-role review=...` override the default
models (any model Inspect supports, e.g. `anthropic/...` or `openrouter/...`).

## Running the cascade

`run_two_stage.py` runs both stages in one go (from the repository root):

```bash
python -m examples.two_stage_judge.run_two_stage -T <transcripts> --scans scans_two_stage \
    --threshold 4 --audit-fraction 0.1
```

1. Triage scans every transcript.
2. Review scans the transcripts with triage score >= `--threshold`, plus a
   random `--audit-fraction` of the rest. The audit sample measures how often
   triage misses something the review finds; use that rate to tune the
   threshold, the triage model and the triage prompt.
3. It writes `two_stage_summary.csv` (one row per transcript: triage and review
   scores, categories and explanations, and why it was reviewed) and prints how
   many escalated transcripts the review confirmed and how many audited ones it
   flagged.

In the results data frames, `value` is the score, `label` the primary category
and `explanation` the justification; the other structured fields (`analysis`,
`issues`, `confidence`, `candidate_issues`) are in `metadata` as JSON.
