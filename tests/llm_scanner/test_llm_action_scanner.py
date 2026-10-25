"""Tests for llm_action_scanner: one judgment per agent action, in context."""

from typing import Callable, Literal

import pytest
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    GenerateConfig,
    Model,
    ModelOutput,
    get_model,
)
from inspect_ai.tool import ToolCall, ToolChoice, ToolInfo
from inspect_scout import Result, Transcript, llm_action_scanner


def _call(call_id: str, cmd: str) -> ToolCall:
    return ToolCall(id=call_id, function="bash", arguments={"cmd": cmd})


def _transcript(padding: str = "") -> Transcript:
    messages: list[ChatMessage] = [
        ChatMessageSystem(content="You are an agent.", id="sys"),
        ChatMessageUser(content="Task: tidy the repo.", id="task"),
        ChatMessageAssistant(
            content="Listing files.", tool_calls=[_call("c1", "ls")], id="a1"
        ),
        ChatMessageTool(
            content="notes.txt tests/" + padding, tool_call_id="c1", id="r1"
        ),
        ChatMessageAssistant(
            content="Removing tests.", tool_calls=[_call("c2", "rm -rf tests")], id="a2"
        ),
        ChatMessageTool(content="removed", tool_call_id="c2", id="r2"),
        ChatMessageAssistant(content="Done.", id="a3"),
    ]
    return Transcript(transcript_id="t", messages=messages)


def _judge(prompts: list[str]) -> Model:
    """Mock judge that records each prompt and flags actions that run `rm`."""

    def respond(
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        prompt = input[-1].text
        prompts.append(prompt)
        action = prompt.split("[BEGIN ACTION]")[-1]
        flagged = "rm -rf" in action
        return ModelOutput.from_content(
            model="mockllm",
            content=f"See [M2]. Explanation.\n\nANSWER: {'yes' if flagged else 'no'}",
        )

    return get_model("mockllm/model", custom_outputs=respond, memoize=False)


async def _run(
    *,
    actions: Literal["tool_calls", "assistant"]
    | Callable[[ChatMessageAssistant], bool] = "tool_calls",
    context: int | None = None,
    include_task: bool = True,
    include_tool_results: bool = False,
    context_window: int | None = None,
    padding: str = "",
) -> tuple[list[Result], list[str]]:
    prompts: list[str] = []
    scanner = llm_action_scanner(
        question="Is this action destructive?",
        answer="boolean",
        model=_judge(prompts),
        actions=actions,
        context=context,
        include_task=include_task,
        include_tool_results=include_tool_results,
        context_window=context_window,
    )
    results = await scanner(_transcript(padding))
    assert isinstance(results, list)
    return results, prompts


@pytest.mark.asyncio
async def test_one_result_per_tool_call_action() -> None:
    results, _ = await _run()

    assert [r.value for r in results] == [False, True]
    assert [r.metadata and r.metadata["action_label"] for r in results] == [
        "M2",
        "M4",
    ]
    assert [r.references[0].id for r in results] == ["a1", "a2"]
    assert all(
        r.metadata and r.metadata["action_tool_calls"] == ["bash"] for r in results
    )


@pytest.mark.parametrize(
    ("actions", "expected_labels"),
    [
        ("tool_calls", ["M2", "M4"]),
        ("assistant", ["M2", "M4", "M6"]),
        (lambda m: not m.tool_calls, ["M6"]),
    ],
)
@pytest.mark.asyncio
async def test_action_selection(
    actions: Literal["tool_calls", "assistant"]
    | Callable[[ChatMessageAssistant], bool],
    expected_labels: list[str],
) -> None:
    results, _ = await _run(actions=actions)

    assert [r.metadata and r.metadata["action_label"] for r in results] == (
        expected_labels
    )


@pytest.mark.asyncio
async def test_prompt_holds_preceding_context_only() -> None:
    _, prompts = await _run()
    second = next(p for p in prompts if "[M4]" in p.split("[BEGIN ACTION]")[-1])
    context, action = second.split("[BEGIN ACTION]")

    assert "Task: tidy the repo." in context
    assert "[M2]" in context and "[M3]" in context
    assert "rm -rf tests" in action
    assert "removed" not in second
    assert "Done." not in second
    assert "You are an agent." not in second


@pytest.mark.asyncio
async def test_include_tool_results_adds_action_outcome() -> None:
    _, prompts = await _run(include_tool_results=True)
    second = next(p for p in prompts if "rm -rf" in p.split("[BEGIN ACTION]")[-1])
    action = second.split("[BEGIN ACTION]")[-1]

    assert "[M5]" in action and "removed" in action


@pytest.mark.parametrize(
    ("include_task", "task_in_context"),
    [(True, True), (False, False)],
)
@pytest.mark.asyncio
async def test_context_limit_keeps_task_and_marks_omission(
    include_task: bool, task_in_context: bool
) -> None:
    _, prompts = await _run(context=1, include_task=include_task)
    second = next(p for p in prompts if "rm -rf" in p.split("[BEGIN ACTION]")[-1])
    context = second.split("[BEGIN ACTION]")[0]

    assert ("Task: tidy the repo." in context) is task_in_context
    assert "[M3]" in context
    assert "[M2]" not in context
    assert "earlier messages omitted" in context or "earlier message omitted" in context


@pytest.mark.asyncio
async def test_small_context_window_drops_oldest_context() -> None:
    # the tool result [M3] is padded past what a 1000-token window leaves
    _, prompts = await _run(
        context_window=1000, include_task=False, padding=" word" * 1000
    )
    second = next(p for p in prompts if "rm -rf" in p.split("[BEGIN ACTION]")[-1])
    context = second.split("[BEGIN ACTION]")[0]

    assert "3 earlier messages omitted" in context
    assert "[M3]" not in context and "[M2]" not in context
    assert "Task: tidy the repo." not in context


@pytest.mark.asyncio
async def test_references_resolve_to_transcript_messages() -> None:
    results, _ = await _run()

    # the judge cites [M2], which must resolve to the same message every time
    assert all(any(ref.id == "a1" for ref in r.references) for r in results)


@pytest.mark.asyncio
async def test_custom_template_variables() -> None:
    prompts: list[str] = []
    scanner = llm_action_scanner(
        template=(
            "{{ action_label }}|{{ action_index }}|{{ task_id }}\n"
            "{{ messages }}\n[BEGIN ACTION]\n{{ action }}\n{{ answer_format }}"
        ),
        answer="boolean",
        model=_judge(prompts),
    )
    await scanner(_transcript())

    assert sorted(p.splitlines()[0] for p in prompts) == ["M2|0|None", "M4|1|None"]


@pytest.mark.asyncio
async def test_no_actions_returns_empty_list() -> None:
    prompts: list[str] = []
    scanner = llm_action_scanner(question="?", answer="boolean", model=_judge(prompts))
    transcript = Transcript(
        transcript_id="t", messages=[ChatMessageUser(content="hi", id="u")]
    )

    assert await scanner(transcript) == []
    assert prompts == []


@pytest.mark.asyncio
async def test_action_too_large_for_context_window_raises() -> None:
    prompts: list[str] = []
    scanner = llm_action_scanner(
        question="?",
        answer="boolean",
        model=_judge(prompts),
        include_tool_results=True,
        context_window=1000,
    )

    with pytest.raises(RuntimeError, match=r"\[M2\].*include_tool_results"):
        await scanner(_transcript(padding=" word" * 1000))


def test_negative_context_raises() -> None:
    with pytest.raises(ValueError, match="context"):
        llm_action_scanner(question="?", answer="boolean", context=-1)
