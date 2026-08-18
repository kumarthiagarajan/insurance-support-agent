import re
from contextlib import contextmanager

from langfuse import Langfuse, get_client, propagate_attributes
from langfuse.langchain import CallbackHandler
from langfuse.types import MaskOtelSpansParams, MaskOtelSpansResult, OtelSpanPatch

_EMAIL_RE = re.compile(r"\b[\w.-]+?@[\w.-]+?\.\w+?\b")
_PHONE_RE = re.compile(r"\b\d{3}[-. ]?\d{3}[-. ]?\d{4}\b")
_CARD_SUFFIX_RE = re.compile(r"\bcard ending \d{4}\b", re.IGNORECASE)


def _redact(text: str) -> str:
    text = _EMAIL_RE.sub("[REDACTED EMAIL]", text)
    text = _PHONE_RE.sub("[REDACTED PHONE]", text)
    text = _CARD_SUFFIX_RE.sub("card ending [REDACTED]", text)
    return text


def _mask_otel_spans(*, params: MaskOtelSpansParams) -> MaskOtelSpansResult | None:
    patches = {}
    for identifier, span in params.spans.items():
        replacements = {}
        for key, value in span.attributes.items():
            if isinstance(value, str):
                masked = _redact(value)
                if masked != value:
                    replacements[key] = masked
        if replacements:
            patches[identifier] = OtelSpanPatch(set_attributes=replacements)
    return MaskOtelSpansResult(span_patches=patches) if patches else None


# Configures the process-wide Langfuse singleton (later get_client() calls reuse it) with
# masking enabled before any tracing happens. Fails open: with no LANGFUSE_* env vars set,
# this logs a warning and the client stays disabled rather than raising, so tracing never
# blocks the app's core support-chat behavior.
Langfuse(mask_otel_spans=_mask_otel_spans)

langfuse = get_client()

THUMBS_SCORE_NAME = "user-thumbs"


@contextmanager
def traced_turn(*, customer_id: str, session_id: str, feature: str, user_message: str):
    """Wrap one graph.invoke() call as a single Langfuse trace.

    Explicitly setting input/output on the root span (rather than letting the
    CallbackHandler's own LangChain run be the root) is what keeps the trace's
    input/output to the user message and assistant reply, instead of the raw
    AgentState dict (internal routing fields, full message history) that
    graph.invoke() actually receives and returns -- see
    https://langfuse.com/docs/observability/best-practices#choose-meaningful-input-and-output.

    `feature` identifies which UI (cli/streamlit/fastapi/gradio) produced the
    trace. Yields (root_span, callback_handler, trace_id); the caller passes
    callback_handler via config={"callbacks": [...]}, calls
    root_span.update(output=...) with this turn's assistant reply/replies
    before the `with` block exits, and holds on to trace_id to attach later
    user feedback via score_turn().
    """
    with langfuse.start_as_current_observation(
        as_type="span", name="support-chat-turn", input=user_message
    ) as root_span:
        with propagate_attributes(
            user_id=customer_id,
            session_id=session_id,
            tags=[feature],
            trace_name="support-chat-turn",
        ):
            # Constructed fresh per turn (matches Langfuse's documented pattern) so it
            # binds to *this* turn's active span context rather than whatever context
            # was active at import time.
            yield root_span, CallbackHandler(), langfuse.get_current_trace_id()


def score_turn(trace_id: str, *, positive: bool, comment: str | None = None) -> None:
    """Record explicit thumbs up/down feedback on a previously traced turn.

    Named for the signal source (a thumbs click), not what it's hoped to
    measure -- see https://langfuse.com/docs/observability/features/user-feedback.
    Uses a deterministic score_id so a user changing their mind (up -> down)
    updates the existing score instead of accumulating duplicates.
    """
    if not trace_id:
        return
    langfuse.create_score(
        trace_id=trace_id,
        name=THUMBS_SCORE_NAME,
        value=1 if positive else 0,
        data_type="BOOLEAN",
        score_id=f"{THUMBS_SCORE_NAME}-{trace_id}",
        comment=comment,
    )
