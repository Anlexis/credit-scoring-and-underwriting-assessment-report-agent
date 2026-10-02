"""AgentCore Platform v1.0"""

# FIN-C2-066 — InputValidateNode
# Inner RAG domain node 1: domain-level validation / normalisation of the
# underwriting query.
#
# Distinct from PreProcessNode (S-1 trust + structural check): this node applies
# domain rules — non-empty, length bounds, whitespace/control normalisation —
# and emits the canonical credit_query consumed by RetrieveNode.
#
# Inner node — ANONYMOUS trust (outer PreProcessNode with VERIFIED_EXTERNAL
# already enforced trust; inner nodes must be ANONYMOUS so the outer
# InvocationContext passes through the GraphNode boundary without rejection;
# review finding 5, corrected 2026-07-02).
#
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
import re
from typing import Any, ClassVar, Dict, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from shared.utils.audit_logger import emit_trace_event

logger = logging.getLogger(__name__)

_MIN_QUERY_CHARS = 3
_MAX_QUERY_CHARS = 4000
_WS_RE = re.compile(r"\s+")


def _normalise_query(raw: str) -> str:
    """Collapse whitespace and trim to a canonical query string."""
    return _WS_RE.sub(" ", raw).strip()


class InputValidateNode(FunctionNode):
    """Domain validation of the underwriting query for FIN-C2-066.

    Inner node — ANONYMOUS trust (see module docstring).

    Input state keys:
        validated_input: str  — normalised query from PreProcessNode
                                (falls back to user_input for unit-test convenience)

    Output state keys (partial dict):
        credit_query: str        — canonical query string
        status:       str
        error_log:    list[str]  (only on ERROR)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        raw = state.get("validated_input") or state.get("user_input", "")

        if not isinstance(raw, str) or not raw.strip():
            emit_trace_event(
                "input_validate_failed",
                {"reason": "empty_query"},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["InputValidateNode: query is empty or missing"],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. "
                + ("InputValidateNode: query is empty or missing"),
            }

        query = _normalise_query(raw)
        if len(query) < _MIN_QUERY_CHARS:
            emit_trace_event(
                "input_validate_failed",
                {"reason": "query_too_short", "length": len(query)},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": [f"InputValidateNode: query too short (<{_MIN_QUERY_CHARS} chars)"],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. "
                + (f"InputValidateNode: query too short (<{_MIN_QUERY_CHARS} chars)"),
            }
        if len(query) > _MAX_QUERY_CHARS:
            # Reject overlength input rather than silently truncating (consistent
            # with PreProcessNode).  Truncation would alter the underwriting query
            # without informing the caller.
            emit_trace_event(
                "input_validate_failed",
                {"reason": "query_too_long", "length": len(query)},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": [f"InputValidateNode: query too long ({len(query)} chars > " f"{_MAX_QUERY_CHARS} limit)"],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. "
                + (f"InputValidateNode: query too long ({len(query)} chars > {_MAX_QUERY_CHARS} limit)"),
            }

        logger.info("InputValidateNode: normalised query_len=%d", len(query))
        emit_trace_event(
            "input_validate_complete",
            {"query_len": len(query)},
            state,
        )

        return {
            "credit_query": query,
            "status": AgentStatus.SUCCESS.value,
        }
