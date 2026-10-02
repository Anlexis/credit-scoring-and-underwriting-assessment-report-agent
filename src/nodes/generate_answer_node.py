"""AgentCore Platform v1.0"""

# FIN-C2-066 — GenerateAnswerNode
# Inner RAG domain node 4 (grounded synthesis): generate the underwriting-
# assessment answer grounded ONLY in the reranked passages, with citations and
# the mandatory advisory disclaimer.
#
# This version performs DETERMINISTIC grounded synthesis from the reranked
# passages — there is NO model call. The llm block in config/config.yaml
# (system_prompt_template / temperature / max_tokens) and the
# prompts/credit_underwriting_assessment.j2 template document the generation
# contract for when real generation is wired in; they are intentionally NOT
# invoked here. This node never fabricates a model call — the grounding contract
# (the answer cites only reranked_documents) holds identically either way.
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

from src.schemas.state import from_json, to_json

logger = logging.getLogger(__name__)

_DISCLAIMER = (
    "Advisory only: this assessment is decision support and does not constitute a "
    "credit decision. A final credit decision requires human underwriter review."
)

_NO_ANSWER = (
    "No underwriting-policy passage met the relevance threshold for this query, so "
    "no grounded assessment can be produced. Please refine the query or escalate to "
    "a human underwriter."
)


def _synthesize_answer(query: str, passages: List[Dict[str, Any]]) -> str:
    """Deterministic grounded synthesis: summarise the cited passages for the query."""
    lines = [
        f"Underwriting assessment for: {query}",
        "",
        "Grounded in the following underwriting-policy passages:",
    ]
    for i, p in enumerate(passages, 1):
        title = p.get("title", p.get("id", "passage"))
        lines.append(f"  [{i}] {title}: {str(p.get('text', '')).strip()}")
    lines.append("")
    lines.append(
        "Assessment: apply the cited policy criteria above to the applicant's data. "
        "Where a threshold is not met, route to the corresponding manual-review path."
    )
    return "\n".join(lines)


class GenerateAnswerNode(FunctionNode):
    """Generate the grounded underwriting-assessment answer.

    Deterministic — no model call; see the module docstring for the generation
    contract, which is documented but not invoked here.

    Inner node — ANONYMOUS trust (see module docstring).

    Input state keys:
        credit_query:       str  — canonical query
        reranked_documents: str  — JSON list of grounding passages (ADR-005)

    Output state keys (partial dict):
        generated_answer: str
        answer_citations: str   — JSON list of {id,title} (ADR-005)
        grounded:         bool
        status:           str
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        query = state.get("credit_query") or state.get("validated_input") or ""
        passages: List[Dict[str, Any]] = from_json(state.get("reranked_documents"), [])

        # Deterministic — no model call. The llm.* settings in config/config.yaml
        # and the Jinja2 prompt template describe the generation wiring only;
        # they are intentionally not read or invoked here.

        if not passages:
            logger.info("GenerateAnswerNode: no grounding passages — no-answer path")
            emit_trace_event(
                "generate_answer_no_grounding",
                {"query_len": len(query)},
                state,
            )
            answer = f"{_NO_ANSWER}\n\n{_DISCLAIMER}"
            return {
                "generated_answer": answer,
                "answer_citations": to_json([]),
                "grounded": False,
                "status": AgentStatus.SUCCESS.value,
            }

        body = _synthesize_answer(query, passages)
        answer = f"{body}\n\n{_DISCLAIMER}"
        citations = [{"id": p.get("id"), "title": p.get("title")} for p in passages]

        logger.info(
            "GenerateAnswerNode: grounded answer chars=%d citations=%d",
            len(answer),
            len(citations),
        )
        emit_trace_event(
            "generate_answer_complete",
            {
                "answer_length": len(answer),
                "citation_count": len(citations),
                "grounded": True,
            },
            state,
        )

        return {
            "generated_answer": answer,
            "answer_citations": to_json(citations),
            "grounded": True,
            "status": AgentStatus.SUCCESS.value,
        }
