"""AgentCore Platform v1.0"""

# FIN-C2-066 — OutputFormatNode (inner domain node 5, last in DomainWorkflowGraph)
# Assemble the final formatted underwriting assessment from generated_answer and
# answer_citations.  This is the last inner node — it produces assessment_report,
# which the outer PostProcessNode S-3-gates before returning to the caller.
#
# Inner node — ANONYMOUS trust (review finding 5, corrected 2026-07-02).
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
from typing import Any, ClassVar, Dict, List, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import from_json
from src.services.source_disclosure import source_label

logger = logging.getLogger(__name__)

_SEPARATOR = "=" * 72
_SUBSEP = "-" * 72


def _assemble(answer: str, citations: List[Dict[str, Any]], grounded: bool) -> str:
    """Assemble the full advisory underwriting-assessment document."""
    lines = [
        _SEPARATOR,
        "CREDIT UNDERWRITING ASSESSMENT (ADVISORY)",
        _SEPARATOR,
        "",
        answer.strip(),
        "",
        _SUBSEP,
        "CITED UNDERWRITING-POLICY PASSAGES",
        _SUBSEP,
    ]
    if citations:
        for i, c in enumerate(citations, 1):
            lines.append(f"  [{i}] {c.get('id', 'N/A')} — {c.get('title', '')}")
    else:
        lines.append("  (none — no passage met the relevance threshold)")
    lines += [
        "",
        _SUBSEP,
        f"Grounded in retrieved policy: {'YES' if grounded else 'NO'}",
        _SEPARATOR,
    ]
    return "\n".join(lines)


class OutputFormatNode(FunctionNode):
    """Assemble the final formatted underwriting assessment (inner domain node).

    Inner node — ANONYMOUS trust (see module docstring).

    Input state keys:
        generated_answer: str  — grounded answer text
        answer_citations: str  — JSON list of {id,title} (ADR-005)
        grounded:         bool

    Output state keys (partial dict):
        assessment_report: str
        result:            str  (same as assessment_report — backbone convention)
        status:            str
        error_log:         list[str]  (only on ERROR)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        answer: str = state.get("generated_answer") or ""
        citations: List[Dict[str, Any]] = from_json(state.get("answer_citations"), [])
        grounded = bool(state.get("grounded"))

        if not answer.strip():
            logger.warning("OutputFormatNode: generated_answer missing in state")
            emit_trace_event(
                "output_format_failed",
                {"reason": "missing_answer"},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["OutputFormatNode: generated_answer missing in state"],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. "
                + ("OutputFormatNode: generated_answer missing in state"),
            }

        report = _assemble(answer, citations, grounded)

        logger.info(
            "OutputFormatNode: assessment chars=%d citations=%d grounded=%s",
            len(report),
            len(citations),
            grounded,
        )
        # Say where the answer came from. This agent answers from a corpus defined inside
        # its own module; a reader seeing a citation has no way to tell that from a live query
        # against the system of record, and the review round rated that confusion its most
        # serious finding. Added before the gate below so it passes the same checks the answer
        # does.
        report = report + source_label(state)
        emit_trace_event(
            "output_format_complete",
            {
                "report_length": len(report),
                "citation_count": len(citations),
                "grounded": grounded,
            },
            state,
        )

        return {
            "assessment_report": report,
            "result": report,
            "status": AgentStatus.SUCCESS.value,
        }
