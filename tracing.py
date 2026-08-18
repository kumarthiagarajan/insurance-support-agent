import re

from langfuse import Langfuse
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


# Configures the process-wide Langfuse singleton (later get_client()/CallbackHandler()
# calls reuse it) with masking enabled before any tracing happens. Fails open: with no
# LANGFUSE_* env vars set, this logs a warning and the client stays disabled rather than
# raising, so tracing never blocks the app's core support-chat behavior.
Langfuse(mask_otel_spans=_mask_otel_spans)

langfuse_handler = CallbackHandler()


def trace_config(*, customer_id: str, session_id: str, feature: str) -> dict:
    """LangGraph invoke() config that attaches Langfuse tracing and trace
    attributes for one conversation turn. `feature` identifies which UI
    (cli/streamlit/fastapi/gradio) produced the trace."""
    return {
        "callbacks": [langfuse_handler],
        "metadata": {
            "langfuse_session_id": session_id,
            "langfuse_user_id": customer_id,
            "langfuse_tags": [feature],
            "langfuse_trace_name": "support-chat-turn",
        },
    }
