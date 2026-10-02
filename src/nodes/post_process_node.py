"""AgentCore Platform v1.0"""

# FIN-C2-066 — PostProcessNode
# Outer backbone post_process slot: credential scan on the assembled assessment
# plus exposure of the final underwriting assessment as formatted_output/result.
#
# Output-gate responsibility: scan the assessment string for credential-like
# patterns before it reaches the caller.  On a violation the node returns a
# sanitised, TRUTHY notice for both formatted_output and result AND clears every
# other state field carrying answer text or a payload, with status=ERROR.
#
# Clearing, not merely replacing, is the point.  LangGraph merges partial deltas,
# so a key the node omits keeps its previous value in state; a downstream reader
# or a checkpoint would then still see the pre-gate assessment the gate refused.
# The delta therefore carries every output-bearing key explicitly, emptied.
#
# The gate is ALSO declared as a method on the agent class
# (CreditScoringUnderwritingAssessmentReportAgent._security_gate_output in
# src/graph/graph.py).  Both call the SAME module-level screen
# (src/security/screens.detect_output_credentials), so the declaration and the
# runtime enforcement cannot drift apart.  Runtime enforcement stays a
# module-level function called from inside execute() — not an instance method on
# the node class, which the real SDK auto-wraps.
#
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
from typing import Any, ClassVar, Dict, Optional, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.security.screens import detect_output_credentials
from src.services.llm_factory import resolve_llm
from src.services.llm_review import render_review, review_result

logger = logging.getLogger(__name__)

# Every state field that carries answer text or a caller-facing payload.  The
# inventory is asserted against the State schema by
# tests/unit/test_nodes.py::test_output_gate_inventory_covers_the_state_schema,
# so a field added later cannot quietly stay out of the cleared set.  `result`
# and `formatted_output` are overwritten with the notice rather than emptied, and
# so are listed separately.
OUTPUT_BEARING_FIELDS: Tuple[str, ...] = (
    "assessment_report",
    "generated_answer",
    "answer_citations",
    "grounded",
)

NOTICE_FIELDS: Tuple[str, ...] = ("formatted_output", "result")


def _security_gate_output(content: str) -> Optional[str]:
    """Scan output for disallowed credential/secret patterns.

    Returns the first violation class name, or None if the output is clean.
    Module-level function (not a node instance method) — this form avoids the
    real-SDK auto-wrap AttributeError on the node class.
    """
    return detect_output_credentials(content)


def _cleared_output_state() -> Dict[str, Any]:
    """Empty values for every output-bearing field, for the violation branch.

    Nothing here is read back out of state: an error envelope that rebuilds
    itself from the state it just cleared is containment in form only.
    """
    return {
        "assessment_report": "",
        "generated_answer": "",
        "answer_citations": "[]",
        "grounded": False,
    }


class PostProcessNode(FunctionNode):
    """Apply the domain output gate and expose the final underwriting assessment.

    Outer backbone post_process slot.  Declared ANONYMOUS — trust was already
    enforced at PreProcessNode (VERIFIED_EXTERNAL).

    Input state keys:
        assessment_report: str  — formatted assessment from inner OutputFormatNode

    Output state keys (partial dict):
        formatted_output:   str
        result:             str
        assessment_report:  str        (cleared on violation)
        generated_answer:   str        (cleared on violation)
        answer_citations:   str        (cleared on violation)
        grounded:           bool       (cleared on violation)
        status:             str
        error_log:          list[str]  (only on ERROR)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        assessment: str = state.get("assessment_report") or state.get("result") or ""

        # ── Fallback for empty assessment ─────────────────────────────────────
        if not assessment.strip():
            logger.warning("PostProcessNode: assessment_report empty — using fallback message")
            assessment = (
                "[Credit Underwriting Assessment] No assessment content generated. "
                "Check error_log for upstream failures."
            )

        # ── Domain output gate ────────────────────────────────────────────────
        _llm, _ = resolve_llm(None, state)
        _remarks = review_result(
            _llm,
            user_input=str(state.get("user_input") or ""),
            result=assessment,
            domain="FIN Credit Scoring & Underwriting Assessment Report Agent",
        )
        _review = render_review(_remarks)
        # Remarks are LLM text derived from the caller's raw words, so they pass through the
        # same gate the answer does -- appending after the gate would put unscanned text past
        # it. A tripped review is dropped on its own: withholding a correct answer because an
        # advisory remark quoted an identifier would let the review change the outcome, and
        # the whole design rests on it being unable to.
        if _review and isinstance(assessment, str) and not _security_gate_output(assessment + _review):
            assessment = assessment + _review

        violation = _security_gate_output(assessment)
        if violation:
            logger.error("PostProcessNode: output gate violation detected — %s", violation)
            emit_trace_event(
                "post_process_s3_violation",
                {"violation": violation},
                state,
            )
            # The notice names the violation CLASS and nothing else: no matched
            # value, no source path, no fragment of the refused assessment.
            sanitised = (
                f"[ASSESSMENT WITHHELD: the assembled assessment matched a disallowed "
                f"pattern ({violation}). Contact the risk/security team.]"
            )
            return {
                **_cleared_output_state(),
                "formatted_output": sanitised,
                "result": sanitised,
                "status": AgentStatus.ERROR.value,
                "error_log": [f"PostProcessNode: output gate refused the assessment — {violation}"],
            }

        logger.info("PostProcessNode: output gate passed — length=%d", len(assessment))
        emit_trace_event(
            "post_process_complete",
            {"output_length": len(assessment)},
            state,
        )

        return {
            "formatted_output": assessment,
            "result": assessment,
            "status": AgentStatus.SUCCESS.value,
        }
