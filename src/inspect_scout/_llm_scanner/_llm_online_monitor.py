from dataclasses import dataclass
from functools import partial
from typing import Any, Awaitable, Callable, Literal

import anyio
from inspect_ai._util._async import tg_collect
from inspect_ai._util.content import ContentText
from inspect_ai.model import (
    CachePolicy,
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
    GenerateConfig,
    Model,
    get_model,
)
from inspect_ai.scorer import ValueToFloat

from .._scanner.extract import (
    MessagesPreprocessor,
    _extract_references,
    check_nonce,
    message_as_str,
)
from .._scanner.result import Reference, Result
from .._scanner.scanner import (
    SCANNER_NAME_ATTR,
    Scanner,
    mark_streaming_support,
    scanner,
)
from .._scanner.util import _message_id
from .._transcript.messages import _effective_segment_budget
from .._transcript.types import Transcript
from ._llm_scanner import _render_template, _resolve_template_kwargs
from .answer import answer_from_argument
from .generate import generate_answer
from .prompt import (
    DEFAULT_ONLINE_MONITOR_TEMPLATE_PREFIX,
    DEFAULT_ONLINE_MONITOR_TEMPLATE_SUFFIX,
)
from .types import AnswerSpec

# Bounds how many per-action prompts are rendered and in flight at once for a
# single transcript (inspect_ai's per-model semaphore still gates connections).
_ACTION_WINDOW_CAP = 16

ActionFilter = Callable[[ChatMessageAssistant], bool]


@dataclass(frozen=True)
class _RenderedMessage:
    label: str
    message: ChatMessage
    text: str


@scanner(messages="all")
def llm_online_monitor(
    *,
    question: str | Callable[[Transcript], Awaitable[str]] | None = None,
    answer: AnswerSpec,
    value_to_float: ValueToFloat | None = None,
    template: str | tuple[str, str] | None = None,
    template_variables: dict[str, Any]
    | Callable[[Transcript], dict[str, Any]]
    | None = None,
    preprocessor: MessagesPreprocessor[list[ChatMessage]] | None = None,
    actions: Literal["tool_calls", "assistant"] | ActionFilter = "tool_calls",
    context: int | None = None,
    include_task: bool = True,
    include_tool_results: bool = False,
    model: str | Model | None = None,
    model_role: str | None = None,
    cache: bool | CachePolicy | None = None,
    retry_refusals: bool | int = 3,
    name: str | None = None,
    context_window: int | None = None,
    nonce: str | None = None,
) -> Scanner[Transcript]:
    """Create a scanner that replays a transcript through a simulated online LLM monitor.

    An online monitor judges an agent while it runs: at each action it sees
    only what has happened so far, and could block the action. This scanner
    simulates that on a collected transcript. Each assistant message that
    counts as an action is judged with only the messages that preceded it,
    never anything after it (by default not even the action's own tool
    results). It is not hindsight analysis: a judge that can see the whole
    transcript (use ``llm_scanner``) can recognize steps as harmful from what
    happened later, which an online monitor cannot.

    Because the replay does not intervene, the transcript continues past
    actions the monitor would have blocked; results after the first such
    action judge steps that a real intervention might have prevented.

    The scanner returns one ``Result`` per action, so each action gets its
    own row in the results data frame, with ``metadata`` identifying the
    action and a reference to the action message first in ``references``.

    Messages are numbered by their position in the transcript (``[M1]``,
    ``[M2]``, ...), so a given message has the same label in every prompt.

    Args:
        question: Question to ask about each action. A static string or a
            function that takes the Transcript and returns a string. Can be
            omitted if you provide a custom template.
        answer: Specification of the answer format (see ``llm_scanner``).
        value_to_float: Optional function to convert the answer value to a float.
        template: Prompt template. Has the variables of ``llm_scanner``
            templates (``{{ messages }}`` is the context followed by the
            action) plus ``{{ context }}`` (the rendered preceding messages),
            ``{{ action }}`` (the rendered action), ``{{ action_label }}``
            (e.g. ``M12``) and ``{{ action_index }}`` (0-based index among the
            transcript's actions). Pass a ``(prefix, suffix)`` tuple to send
            the prompt as two content blocks with ``cache_prompt=True``.
            Defaults to a template that puts the context first, so prompts
            for successive actions share a cacheable prefix.
        template_variables: Additional variables to make available in the
            template. Optionally takes a function which receives the current
            ``Transcript`` and returns variables.
        preprocessor: Transform messages before analysis and control which
            content is rendered (system messages, reasoning, tool usage).
            Defaults to excluding system messages. A ``transform`` receives
            the transcript's full message list once.
        actions: Which assistant messages to judge: ``"tool_calls"`` (default)
            for assistant messages that make tool calls, ``"assistant"`` for
            every assistant message, or a function that takes a
            ``ChatMessageAssistant`` and returns whether to judge it.
        context: Number of messages preceding each action to include as
            context. ``None`` (default) includes all preceding messages.
        include_task: Always include the first user message (the task) in
            the context, even when older context is dropped by ``context``
            or to fit the context window.
        include_tool_results: Include the tool results answering the
            action's tool calls in the action block. Defaults to ``False``,
            which judges the action as it was proposed, before it ran.
        model: Optional model specification. Can be a model name string or
            ``Model`` instance. If None, uses the default model.
        model_role: Optional model role for role-based model resolution.
        cache: Response caching policy for the judge model call.
        retry_refusals: Retry model refusals. Pass an ``int`` for number of
            retries (defaults to 3). Pass ``False`` to not retry refusals.
        name: Scanner name.
        context_window: Override the model's context window size. When a
            prompt would not fit, the oldest context messages are dropped
            (keeping the task if ``include_task``), and an omission marker
            is written in their place.
        nonce: A code written on every marker in the rendered messages and
            available to the template as ``{{ scout_nonce }}`` (see
            ``llm_scanner``). Make it with ``secrets.token_hex(4)``.

    Returns:
        A ``Scanner`` that analyzes a Transcript and returns a list with one
        ``Result`` per action (empty if the transcript has no actions).

    Raises:
        ValueError: If ``context`` is negative or ``nonce`` is malformed.
    """
    check_nonce(nonce)
    if context is not None and context < 0:
        raise ValueError(f"context must be None or >= 0, got {context}")

    resolved_template: str | tuple[str, str] = (
        (DEFAULT_ONLINE_MONITOR_TEMPLATE_PREFIX, DEFAULT_ONLINE_MONITOR_TEMPLATE_SUFFIX)
        if template is None
        else template
    )
    resolved_answer = answer_from_argument(answer)
    resolved_retry_refusals = (
        retry_refusals
        if isinstance(retry_refusals, int)
        else 3
        if retry_refusals is True
        else 0
    )
    is_action = _action_filter(actions)

    async def scan(transcript: Transcript) -> list[Result]:
        resolved_model = get_model(model, role=model_role)

        messages = transcript.messages
        if preprocessor is not None and preprocessor.transform is not None:
            messages = await preprocessor.transform(messages)

        rendered = _render_messages(messages, preprocessor, nonce)
        action_positions = [
            pos
            for pos, r in enumerate(rendered)
            if isinstance(r.message, ChatMessageAssistant) and is_action(r.message)
        ]
        if not action_positions:
            return []

        id_map = {r.label: _message_id(r.message) for r in rendered}
        task_position = next(
            (
                pos
                for pos, r in enumerate(rendered)
                if isinstance(r.message, ChatMessageUser)
            ),
            None,
        )

        base_kwargs = await _resolve_template_kwargs(
            template_variables=template_variables,
            transcript=transcript,
            messages="",
            question=question,
            answer=resolved_answer,
            nonce=nonce,
        )
        budget = _effective_segment_budget(
            model=resolved_model,
            context_window=context_window,
            prompt_reserve=await resolved_model.count_tokens(
                "".join(
                    _render_prompt_text(resolved_template, base_kwargs, "", "", "", 0)
                )
            ),
        )
        token_sizes = await _token_sizes(rendered, resolved_model, budget)

        window = anyio.Semaphore(_ACTION_WINDOW_CAP)

        async def judge(action_index: int, position: int) -> Result:
            async with window:
                action_message = rendered[position].message
                action_positions_in_block = [position] + (
                    _tool_result_positions(rendered, position)
                    if include_tool_results
                    else []
                )
                action_tokens = sum(token_sizes[p] for p in action_positions_in_block)
                if action_tokens > budget:
                    raise RuntimeError(
                        f"The action [{rendered[position].label}] "
                        f"({action_tokens} tokens) does not fit in the "
                        f"{budget} tokens left by the context window after the "
                        "prompt template. Increase context_window"
                        + (
                            " or set include_tool_results=False."
                            if include_tool_results
                            else "."
                        )
                    )
                context_positions = _fit_context(
                    position=position,
                    context=context,
                    task_position=task_position if include_task else None,
                    sizes=token_sizes,
                    budget=budget - action_tokens,
                )
                context_str = _join_with_omissions(
                    rendered, context_positions, position
                )
                action_str = "\n".join(
                    _numbered(rendered[p]) for p in action_positions_in_block
                )
                prompt, config = _build_prompt(
                    resolved_template,
                    base_kwargs,
                    context_str,
                    action_str,
                    rendered[position].label,
                    action_index,
                    cache,
                )
                result = await generate_answer(
                    prompt,
                    answer,
                    model=resolved_model,
                    config=config,
                    retry_refusals=resolved_retry_refusals,
                    extract_refs=lambda text: _extract_references(text, id_map),
                    value_to_float=value_to_float,
                )
                return _with_action_info(
                    result,
                    action_message,
                    rendered[position].label,
                    action_index,
                )

        return await tg_collect(
            [partial(judge, i, p) for i, p in enumerate(action_positions)]
        )

    if name is not None:
        setattr(scan, SCANNER_NAME_ATTR, name)

    # Each action needs every message before it, so the scanner always takes
    # a materialized transcript.
    mark_streaming_support(scan, False)

    return scan


def _action_filter(
    actions: Literal["tool_calls", "assistant"] | ActionFilter,
) -> ActionFilter:
    if actions == "tool_calls":
        return lambda message: bool(message.tool_calls)
    if actions == "assistant":
        return lambda message: True
    if callable(actions):
        return actions
    raise ValueError(
        f"actions must be 'tool_calls', 'assistant' or a function, got {actions!r}"
    )


def _render_messages(
    messages: list[ChatMessage],
    preprocessor: MessagesPreprocessor[list[ChatMessage]] | None,
    nonce: str | None,
) -> list[_RenderedMessage]:
    rendered: list[_RenderedMessage] = []
    for message in messages:
        text = message_as_str(message, preprocessor, nonce=nonce)
        if text is not None:
            rendered.append(
                _RenderedMessage(
                    label=f"M{len(rendered) + 1}", message=message, text=text
                )
            )
    return rendered


def _numbered(message: _RenderedMessage) -> str:
    return f"[{message.label}] {message.text}"


def _utf8_len(text: str) -> int:
    return len(text.encode("utf-8"))


async def _token_sizes(
    rendered: list[_RenderedMessage], model: Model, budget: int
) -> list[int]:
    """Token size of each rendered message (as numbered in prompts).

    UTF-8 byte length bounds the token count from above for byte-level
    tokenizers, so the model's tokenizer is only consulted when the whole
    transcript might not fit in the budget.
    """
    byte_sizes = [_utf8_len(_numbered(r)) + 1 for r in rendered]
    if sum(byte_sizes) <= budget:
        return byte_sizes
    return await tg_collect(
        [partial(model.count_tokens, _numbered(r) + "\n") for r in rendered]
    )


def _tool_result_positions(
    rendered: list[_RenderedMessage], position: int
) -> list[int]:
    message = rendered[position].message
    call_ids = {
        call.id
        for call in (
            message.tool_calls if isinstance(message, ChatMessageAssistant) else None
        )
        or []
    }
    positions: list[int] = []
    for pos in range(position + 1, len(rendered)):
        candidate = rendered[pos].message
        if not isinstance(candidate, ChatMessageTool):
            break
        if candidate.tool_call_id in call_ids:
            positions.append(pos)
    return positions


def _fit_context(
    *,
    position: int,
    context: int | None,
    task_position: int | None,
    sizes: list[int],
    budget: int,
) -> list[int]:
    """Positions of the context messages for the action at `position`.

    Keeps the task message when `task_position` is given, precedes the action
    and fits in `budget`, then the most recent messages (up to `context`) that
    fit in what remains.
    """
    pinned = (
        [task_position]
        if task_position is not None
        and task_position < position
        and sizes[task_position] <= budget
        else []
    )
    remaining = budget - sum(sizes[p] for p in pinned)
    first = 0 if context is None else max(0, position - context)
    recent: list[int] = []
    for pos in range(position - 1, first - 1, -1):
        if pos in pinned:
            continue
        if sizes[pos] > remaining:
            break
        remaining -= sizes[pos]
        recent.append(pos)
    return sorted(pinned + recent)


def _join_with_omissions(
    rendered: list[_RenderedMessage], positions: list[int], action_position: int
) -> str:
    """Render the context messages, marking each run of dropped messages."""
    parts: list[str] = []
    expected = 0
    for pos in positions:
        if pos > expected:
            parts.append(_omission(pos - expected))
        parts.append(_numbered(rendered[pos]))
        expected = pos + 1
    if action_position > expected:
        parts.append(_omission(action_position - expected))
    return "\n".join(parts)


def _omission(count: int) -> str:
    noun = "message" if count == 1 else "messages"
    return f"[... {count} earlier {noun} omitted ...]\n"


def _render_prompt_text(
    template: str | tuple[str, str],
    base_kwargs: dict[str, Any],
    context_str: str,
    action_str: str,
    action_label: str,
    action_index: int,
) -> list[str]:
    kwargs = {
        **base_kwargs,
        "messages": f"{context_str}\n{action_str}" if context_str else action_str,
        "context": context_str,
        "action": action_str,
        "action_label": action_label,
        "action_index": action_index,
    }
    templates = template if isinstance(template, tuple) else (template,)
    return [_render_template(t, kwargs) for t in templates]


def _build_prompt(
    template: str | tuple[str, str],
    base_kwargs: dict[str, Any],
    context_str: str,
    action_str: str,
    action_label: str,
    action_index: int,
    cache: bool | CachePolicy | None,
) -> tuple[str | list[ChatMessage], GenerateConfig | None]:
    parts = _render_prompt_text(
        template, base_kwargs, context_str, action_str, action_label, action_index
    )
    if isinstance(template, tuple):
        return (
            [ChatMessageUser(content=[ContentText(text=part) for part in parts])],
            GenerateConfig(cache_prompt=True, cache=cache),
        )
    return parts[0], GenerateConfig(cache=cache) if cache is not None else None


def _with_action_info(
    result: Result,
    action_message: ChatMessage,
    action_label: str,
    action_index: int,
) -> Result:
    action_id = _message_id(action_message)
    references = [
        Reference(type="message", cite=f"[{action_label}]", id=action_id),
        *(ref for ref in result.references if ref.id != action_id),
    ]
    tool_calls = (
        [call.function for call in action_message.tool_calls or []]
        if isinstance(action_message, ChatMessageAssistant)
        else []
    )
    metadata = {
        **(result.metadata or {}),
        "action_index": action_index,
        "action_label": action_label,
        "action_message_id": action_id,
        "action_tool_calls": tool_calls,
    }
    return result.model_copy(update={"references": references, "metadata": metadata})
