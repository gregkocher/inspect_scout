# Additions in this fork

This fork of [Inspect Scout](https://github.com/meridianlabs-ai/inspect_scout)
adds the following on top of upstream `main`.

## Per-action LLM scanning: `llm_action_scanner()`

Judges each agent action with an LLM, in the context that preceded it, and
returns one result per action (where `llm_scanner()` returns one result per
transcript). By default an action is an assistant message that makes tool
calls.

```python
from inspect_scout import llm_action_scanner

llm_action_scanner(
    question="How suspicious is this action? 0 = benign, 10 = malicious.",
    answer="numeric",
    context=20,
)
```

- Guide, options and result columns: [examples/action_scanner/README.md](examples/action_scanner/README.md)
- Runnable example scanners: [examples/action_scanner/action_scanners.py](examples/action_scanner/action_scanners.py)
- Reference docs section: "Scanning Actions" in [docs/llm_scanner.qmd](docs/llm_scanner.qmd)
- Code: [src/inspect_scout/_llm_scanner/_llm_action_scanner.py](src/inspect_scout/_llm_scanner/_llm_action_scanner.py)
- Tests: [tests/llm_scanner/test_llm_action_scanner.py](tests/llm_scanner/test_llm_action_scanner.py)

## Importing OpenAI Chat Completions trajectories

A script that converts a folder of OpenAI Chat Completions trajectories into a
transcript database, normalizing common format deviations and checking that
nothing is lost in the conversion.

- Guide: [examples/sources/openai_chat/README.md](examples/sources/openai_chat/README.md)
- Script: [examples/sources/openai_chat/openai_to_scout.py](examples/sources/openai_chat/openai_to_scout.py)

## Installing this fork

```bash
git clone https://github.com/gregkocher/inspect_scout
cd inspect_scout
git checkout feat/llm-action-scanner
python3 -m venv .venv && .venv/bin/pip install -e .
```
