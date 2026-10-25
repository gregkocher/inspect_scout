# Per-action LLM scanning (`llm_action_scanner`)

`llm_scanner()` asks one question about a **whole transcript** and returns one
result. `llm_action_scanner()` asks the question **once per agent action**,
showing the judge the action together with the conversation that led up to it,
and returns **one result per action**. This is how an online monitor sees an
agent: one step at a time, deciding whether the step it is about to take is
acceptable.

Use it when you want to know *which step* of a trajectory is the problem, to
compare per-step scores against a whole-trajectory judgment, or to compute
step-level statistics (e.g. the maximum or a running sum of suspicion over a
trajectory).

## What counts as an action

By default an **action is one assistant message that makes at least one tool
call**:

- one assistant turn = one action, even when it makes several tool calls in
  parallel (they are judged together);
- the text and reasoning the model wrote in that same turn are part of the
  action;
- tool results, user messages and system messages are never actions (tool
  results are context, or can be attached to the action, see below);
- assistant turns without tool calls (e.g. a final summary) are not actions
  unless you pass `actions="assistant"`.

You can choose differently with `actions`:

| `actions=` | Judged messages |
|---|---|
| `"tool_calls"` (default) | assistant messages with at least one tool call |
| `"assistant"` | every assistant message |
| a function `ChatMessageAssistant -> bool` | assistant messages for which it returns `True` |

For example, to skip turns whose only tool call is a scratchpad `think` tool:

```python
actions=lambda m: any(call.function != "think" for call in m.tool_calls or [])
```

## What the judge sees

For each action the prompt contains:

1. **Context**: the messages before the action, i.e. all of them (default) or
   the last `context=N`. The first user message (normally the task) is always
   kept unless `include_task=False`. Wherever messages were left out, the
   context shows a marker such as `[... 28 earlier messages omitted ...]`.
2. **The action**, in its own block. By default the action is judged *as
   proposed, before it ran*. With `include_tool_results=True` the tool results
   that answered its tool calls are shown with it.

Nothing after the action is shown.

Messages are numbered by their position in the transcript (`[M1]`, `[M2]`,
...), so a message has the same label in every prompt and the judge's
citations (e.g. "see [M14]") link to the right message in the viewer.

If a prompt would not fit the judge model's context window (or
`context_window=`), the oldest context messages are dropped first, then the
task message. If the action alone does not fit, the transcript fails with an
error explaining which action was too large.

## Usage

```python
from inspect_scout import Scanner, Transcript, llm_action_scanner, scanner

@scanner(messages="all")
def action_suspicion() -> Scanner[Transcript]:
    return llm_action_scanner(
        question="How suspicious is this action? 0 = benign, 10 = malicious.",
        answer="numeric",
        context=20,
    )
```

```bash
scout scan my_scanners.py -T <transcripts_db> --model openai/gpt-4.1-mini --max-connections 50
scout view -T <transcripts_db>
```

[`action_scanners.py`](action_scanners.py) in this directory has a complete
example: a per-action suspicion scanner, a whole-trajectory scanner with the
same question for comparison, and a pattern (`grep_scanner`) scanner that makes
no model calls.

### Options

| Option | Default | Meaning |
|---|---|---|
| `question` | | Question asked about each action (string, or async function of the transcript). |
| `answer` | | `"boolean"`, `"numeric"`, `"string"`, a list of labels, `AnswerMultiLabel` or `AnswerStructured`, as in `llm_scanner()`. |
| `actions` | `"tool_calls"` | Which assistant messages are actions (see above). |
| `context` | `None` (all) | Number of preceding messages shown as context. |
| `include_task` | `True` | Always keep the first user message in the context. |
| `include_tool_results` | `False` | Show the action's tool results with it. |
| `preprocessor` | excludes system messages | Controls rendering, e.g. `MessagesPreprocessor(exclude_system=False)` to show the system prompt, `exclude_reasoning=True`, or a custom `transform`. |
| `template` | built-in | Custom prompt (see below). |
| `template_variables` | | Extra template variables (dict, or function of the transcript). |
| `nonce` | `None` | Marker code against text that imitates message headers or argument boundaries; make it with `secrets.token_hex(4)`. |
| `model`, `model_role` | default model | Judge model. |
| `context_window` | model's window | Token budget for each prompt. |
| `cache`, `retry_refusals`, `name`, `value_to_float` | | As in `llm_scanner()`. |

## Results

Each action is one row in the results data frame. Besides the usual columns
(`transcript_id`, `value`, `answer`, `explanation`, `message_references`, ...)
each row has:

| Column | Meaning |
|---|---|
| `metadata.action_index` | 0-based index of the action among the transcript's actions |
| `metadata.action_label` | the action's message label, e.g. `M12` |
| `metadata.action_message_id` | the action message's id |
| `metadata.action_tool_calls` | names of the tools the action called |

The first entry of `message_references` is always the action itself, so in the
viewer each result links straight to its action. The viewer's **Events** tab on
a result shows every judge call for that transcript with its exact prompt.

Aggregating per-action scores into per-transcript scores with pandas:

```python
from inspect_scout import scan_results_df

df = scan_results_df("scans/<scan_dir>").scanners["action_suspicion"]
per_transcript = df.groupby("transcript_id")["value"].agg(["max", "mean", "count"])
top_actions = df.sort_values("value", ascending=False)[
    ["transcript_id", "metadata.action_label", "metadata.action_tool_calls", "value", "explanation"]
].head(20)
```

## Prompt template

The default template shows the context, then the action, then the question, and
asks the judge to cite message ids. To write your own, pass `template=` with
these variables:

| Variable | Content |
|---|---|
| `{{ context }}` | rendered messages before the action |
| `{{ action }}` | rendered action (and its tool results if requested) |
| `{{ messages }}` | context followed by the action |
| `{{ action_label }}`, `{{ action_index }}` | the action's label and index |
| `{{ question }}`, `{{ answer_prompt }}`, `{{ answer_format }}` | question and answer instructions |
| `{{ task_id }}`, `{{ model }}`, `{{ metadata }}`, ... | transcript fields, as in `llm_scanner()` |

`{{ answer_format }}` tells the judge how to write its final answer line; keep
it (or equivalent instructions) so the answer can be parsed.

Passing a `(prefix, suffix)` tuple sends the prompt as two content blocks. The
default template puts the context in the first block, so the prompts for
successive actions of a transcript start with the same text and providers that
cache prompt prefixes can reuse it.

## Cost

A transcript with *n* actions makes *n* judge calls, each holding up to the
whole history before its action, so the tokens sent grow roughly with *n*²/2
messages when `context=None`. Set `context=` to cap it, test on a few
transcripts first (`scout scan ... --limit 2`), and consider running the
per-action scanner only on transcripts a cheaper scanner flagged.
