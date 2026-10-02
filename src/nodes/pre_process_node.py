"""AgentCore Platform v1.0"""

# FIN-C2-066 — PreProcessNode
# Outer backbone pre_process slot (S-1 trust gate + query validation).
#
# Responsibilities:
#   - Enforce VERIFIED_EXTERNAL trust (required_trust_level = S-1 gate)
#   - Screen the caller payload for chat-template control tokens (fail closed)
#   - Mask the personal data the framework's boundary-delimited patterns cannot
#     see inside unspaced Japanese text
#   - Reject empty input early (fail-fast)
#   - Reject overlength input with a validation error (never silently truncate)
#   - Reject a malformed structured payload without echoing it back
#   - Accept either a plain question string or a JSON object carrying a query
#   - Write validated_input (normalised query) + enriched_context to State
#   - Emit an S-4 audit event for every validation decision
#
# Returns ONLY the state keys this node writes (partial-dict contract).

import json
import logging
from typing import Any, ClassVar, Dict, Optional, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import to_json
from src.security.screens import mask_payload_pii, screen_control_tokens

logger = logging.getLogger(__name__)

_MAX_QUERY_CHARS = 4000

# Closed set of rejection reasons.  A rejection names the condition, never the
# value that caused it, so no caller-supplied text is echoed back on refusal.
_REASON_EMPTY = "empty_input"
_REASON_MALFORMED = "malformed_payload"
_REASON_NO_QUERY = "no_query"
_REASON_TOO_LONG = "query_too_long"
_REASON_CONTROL_TOKEN = "chat_template_control_token"
_REASON_UNMASKABLE_PII = "unmaskable_personal_data"


def _extract_query(raw: str) -> Tuple[str, str]:
    """Return ``(query, reason)`` for a plain string or a JSON object payload.

    ``reason`` is the empty string on success.  A payload that opens like JSON
    but does not parse is REJECTED rather than passed through as its own raw
    text: the framework's S-2 gate masks personal data in ``user_input`` before
    this node ever parses it, and a mask landing inside a ``\\u`` escape leaves
    the payload malformed.  Returning the raw envelope in that case put the
    caller's escaped JSON — brace, escapes and all — into the rendered
    assessment as if it were their question.
    """
    stripped = raw.strip()
    if stripped[:1] not in ("{", "["):
        return stripped, ""
    try:
        payload = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return "", _REASON_MALFORMED
    if not isinstance(payload, dict):
        return "", _REASON_MALFORMED
    for key in ("query", "question", "prompt", "text"):
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip(), ""
    return "", _REASON_NO_QUERY


class PreProcessNode(FunctionNode):
    """S-1 input validation for FIN-C2-066.

    Validates the caller-supplied underwriting query before the domain
    workflow runs.  This is the outer backbone's pre_process slot — the only
    node with VERIFIED_EXTERNAL trust so that unauthenticated or anonymous
    callers are rejected here (fail-fast; inner domain nodes carry ANONYMOUS
    trust and never see untrusted input directly).

    It is also the single caller-data entry boundary, so the domain input
    screens hang off this node and nowhere else: one layer, so a mutant that
    removes it is observable end-to-end.

    Input state keys:
        user_input: str  — caller-supplied question (plain text or JSON w/ query)

    Output state keys (partial dict):
        validated_input:  str        — normalised query string
        enriched_context: str        — JSON-serialised channel metadata
        status:           str        — AgentStatus.SUCCESS or ERROR
        error_log:        list[str]  — set only on ERROR
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def _extra_security_gate_input(self, state: AgentState) -> AgentState:
        """Domain input screen — runs after the framework's PII/injection pass.

        Two things the framework's defaults do not cover:

        * **Chat-template control tokens as a class.**  The framework blocks the
          literal ``<|im_start|>``/``<|im_end|>`` and the bracketed ``[INST]``
          forms; ``<<SYS>>`` and every other ``<|…|>`` token pass it.  Screened
          raw, markup-stripped, and — when the payload is JSON — over every
          parsed string leaf and mapping key, so an escaped token cannot survive
          parsing unseen.  Fail closed.
        * **Personal data in unspaced Japanese.**  The framework's digit-group
          patterns are word-boundary delimited and Kana/Kanji are word
          characters, so ``個人番号1234-5678-9012を確認`` yields no findings while
          the ASCII-spaced form masks.  The residual screen re-runs the
          framework's own detector across the script boundary and masks what it
          reports, which is what keeps the two block sets identical.  It runs
          over the parsed payload as well as the raw text, because a client
          library's default JSON encoding turns Japanese into ``\\uXXXX``
          escapes and an all-ASCII envelope has no script boundary to find.

        Contract: never raises; a refusal is surfaced through state.
        """
        screened: Dict[str, Any] = dict(state)
        raw = screened.get("user_input", "")
        if not isinstance(raw, str) or not raw:
            return screened

        token_class = screen_control_tokens(raw)
        if token_class:
            emit_trace_event(
                "pre_process_control_token_blocked",
                {"token_class": token_class},
                screened,
            )
            screened["status"] = AgentStatus.ERROR.value
            screened["error_log"] = [f"PreProcessNode: input rejected — {_REASON_CONTROL_TOKEN}"]
            return screened

        masked, masked_types, unmaskable_types = mask_payload_pii(raw)
        if unmaskable_types:
            emit_trace_event(
                "pre_process_residual_pii_unmaskable",
                {"types": unmaskable_types},
                screened,
            )
            screened["status"] = AgentStatus.ERROR.value
            screened["error_log"] = [f"PreProcessNode: input rejected — {_REASON_UNMASKABLE_PII}"]
            return screened
        if masked != raw:
            emit_trace_event(
                "pre_process_residual_pii_masked",
                {"types": masked_types},
                screened,
            )
            screened["user_input"] = masked
        return screened

    def execute(self, state: AgentState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        user_input = state.get("user_input", "")
        input_context = state.get("input_context", {}) or {}

        # ── Emptiness check ───────────────────────────────────────────────────
        if not user_input or not isinstance(user_input, str) or not user_input.strip():
            logger.warning("PreProcessNode: user_input is empty or missing")
            emit_trace_event(
                "pre_process_validation_failed",
                {"reason": _REASON_EMPTY},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["PreProcessNode: user_input is empty or missing"],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. "
                + ("PreProcessNode: user_input is empty or missing"),
            }

        # ── Query extraction ──────────────────────────────────────────────────
        query, reason = _extract_query(user_input)
        if reason == _REASON_MALFORMED:
            logger.warning("PreProcessNode: structured payload did not parse — rejected")
            emit_trace_event(
                "pre_process_validation_failed",
                {"reason": _REASON_MALFORMED},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": [
                    "PreProcessNode: structured payload is not a JSON object with a "
                    "query field — resubmit a well-formed payload"
                ],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. "
                + (
                    "PreProcessNode: structured payload is not a JSON object with a query field — resubmit a well-formed payload"
                ),
            }
        if not query:
            emit_trace_event(
                "pre_process_validation_failed",
                {"reason": _REASON_NO_QUERY},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["PreProcessNode: no underwriting query found in input"],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. "
                + ("PreProcessNode: no underwriting query found in input"),
            }

        # ── Overlength check ──────────────────────────────────────────────────
        # Reject overlength input with a validation error instead of silently
        # truncating it.  Truncation would alter an underwriting-assessment
        # request without informing the caller and could produce an incomplete
        # decision-support result.
        if len(query) > _MAX_QUERY_CHARS:
            logger.warning(
                "PreProcessNode: query too long (%d > %d chars) — rejected",
                len(query),
                _MAX_QUERY_CHARS,
            )
            emit_trace_event(
                "pre_process_validation_failed",
                {"reason": _REASON_TOO_LONG, "length": len(query)},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": [
                    f"PreProcessNode: query too long ({len(query)} chars > "
                    f"{_MAX_QUERY_CHARS} limit) — resubmit a shorter query"
                ],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. "
                + (
                    f"PreProcessNode: query too long ({len(query)} chars > {_MAX_QUERY_CHARS} limit) — resubmit a shorter query"
                ),
            }

        logger.info("PreProcessNode: validated query_len=%d", len(query))
        emit_trace_event(
            "pre_process_validated",
            {"query_len": len(query)},
            state,
        )

        return {
            "validated_input": query,
            "enriched_context": to_json(
                {
                    "source": "CreditScoringUnderwritingAssessmentReportAgent",
                    "channel": input_context.get("channel", "unknown"),
                }
            ),
            "status": AgentStatus.SUCCESS.value,
        }
