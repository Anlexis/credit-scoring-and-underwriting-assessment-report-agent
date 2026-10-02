"""AgentCore Platform v1.0"""

# FIN-C2-066 — RerankFilterNode
# Inner RAG domain node 3: rerank the retrieved passages and filter by the
# score_threshold, keeping the most relevant grounding passages.
#
# v1: deterministic re-scoring on the retrieval score; production can swap in a
# cross-encoder reranker.  An empty retrieval set is a valid no-answer state
# (not an error).
#
# Runtime config: score_threshold and rerank_keep come from config/config.yaml
# (retrieval.*), forwarded by the outer _parent_config() and seeded — finite and
# bounded — into inner state by DomainWorkflowGraph._extra_initial_state().
# Read precedence: state (real .invoke() path) -> config["configurable"]
# (direct unit calls) -> node default.
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

_DEFAULT_SCORE_THRESHOLD = 0.1
_DEFAULT_KEEP = 3


class RerankFilterNode(FunctionNode):
    """Rerank and threshold-filter the retrieved underwriting passages.

    Inner node — ANONYMOUS trust (see module docstring).

    Input state keys:
        retrieved_documents: str    — JSON list of {id,title,text,score} (ADR-005)
        score_threshold:     float  — seeded from config/config.yaml (optional)
        rerank_keep:         int    — seeded from config/config.yaml (optional)

    Output state keys (partial dict):
        reranked_documents: str  — JSON list of kept passages (ADR-005)
        status:             str
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        retrieved: List[Dict[str, Any]] = from_json(state.get("retrieved_documents"), [])

        # Config precedence: declared values seeded into inner state by
        # DomainWorkflowGraph._extra_initial_state() (real .invoke() path) ->
        # config["configurable"] (direct unit calls) -> node default.
        configurable = (config or {}).get("configurable", {})
        threshold_val = state.get("score_threshold")
        if threshold_val is None:
            threshold_val = configurable.get("score_threshold", _DEFAULT_SCORE_THRESHOLD)
        score_threshold = float(threshold_val)
        keep_val = state.get("rerank_keep")
        if keep_val is None:
            keep_val = configurable.get("rerank_keep", _DEFAULT_KEEP)
        keep = int(keep_val)

        if not retrieved:
            # No retrieval hits is a valid state (no-answer path) — not an error.
            emit_trace_event(
                "rerank_filter_empty",
                {"reason": "no_retrieved_documents"},
                state,
            )
            return {
                "reranked_documents": to_json([]),
                "status": AgentStatus.SUCCESS.value,
            }

        # Rerank (descending score), threshold-filter, then keep the top-N.
        ranked = sorted(retrieved, key=lambda d: float(d.get("score", 0.0)), reverse=True)
        filtered = [d for d in ranked if float(d.get("score", 0.0)) >= score_threshold]
        reranked = filtered[:keep]

        logger.info(
            "RerankFilterNode: in=%d filtered=%d kept=%d threshold=%.3f",
            len(retrieved),
            len(filtered),
            len(reranked),
            score_threshold,
        )
        emit_trace_event(
            "rerank_filter_complete",
            {
                "input_count": len(retrieved),
                "filtered_count": len(filtered),
                "kept_count": len(reranked),
                "score_threshold": score_threshold,
            },
            state,
        )

        return {
            "reranked_documents": to_json(reranked),
            "status": AgentStatus.SUCCESS.value,
        }
