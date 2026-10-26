# Additions in this fork

This fork of [Inspect Scout](https://github.com/meridianlabs-ai/inspect_scout)
adds the following on top of upstream `main`.

## Simulated online monitoring: `llm_online_monitor()`

Replays a collected transcript through a simulated online (real-time) LLM
monitor: each agent action is judged with only the messages before it, as a
monitor watching the agent live would see it, and the scanner returns one
result per action. This is not after-the-fact analysis of the full transcript
(that is `llm_scanner()`, whose judge also sees what happened later). By
default an action is an assistant message that makes tool calls.

```python
from inspect_scout import llm_online_monitor

llm_online_monitor(
    question="How suspicious is this action? 0 = benign, 10 = malicious.",
    answer="numeric",
    context=20,
)
```

- Guide, options and result columns: [examples/online_monitor/README.md](examples/online_monitor/README.md)
- Runnable example scanners: [examples/online_monitor/online_monitor_scanners.py](examples/online_monitor/online_monitor_scanners.py)
- Reference docs section: "Simulating an Online Monitor" in [docs/llm_scanner.qmd](docs/llm_scanner.qmd)
- Code: [src/inspect_scout/_llm_scanner/_llm_online_monitor.py](src/inspect_scout/_llm_scanner/_llm_online_monitor.py)
- Tests: [tests/llm_scanner/test_llm_online_monitor.py](tests/llm_scanner/test_llm_online_monitor.py)

## Importing OpenAI Chat Completions trajectories

A script that converts a folder of OpenAI Chat Completions trajectories into a
transcript database, normalizing common format deviations and checking that
nothing is lost in the conversion.

- Guide: [examples/sources/openai_chat/README.md](examples/sources/openai_chat/README.md)
- Script: [examples/sources/openai_chat/openai_to_scout.py](examples/sources/openai_chat/openai_to_scout.py)

## Two-stage LLM judging

Two whole-transcript scanners meant as a cascade: a cheap, high-recall triage
scanner and a careful review scanner with a calibrated 1-10 severity rubric,
plus a script that runs triage on everything and review on the escalated
transcripts and a random audit sample of the rest.

- Guide: [examples/two_stage_judge/README.md](examples/two_stage_judge/README.md)
- Scanners: [examples/two_stage_judge/two_stage_scanners.py](examples/two_stage_judge/two_stage_scanners.py)
- Cascade script: [examples/two_stage_judge/run_two_stage.py](examples/two_stage_judge/run_two_stage.py)

## Installing this fork

```bash
git clone https://github.com/gregkocher/inspect_scout
cd inspect_scout
git checkout feat/llm-online-monitor
python3 -m venv .venv && .venv/bin/pip install -e .
```
