"""AgentCore Platform v1.0"""

# ADR-005: State must be a flat TypedDict — never Pydantic BaseModel.
# LangGraph checkpoints use msgpack serialization; Pydantic objects
# cause silent corruption.  Extend AgentState with agent-specific
# fields only.  Do NOT add credentials, secrets, or Pydantic models.
#
# FIN-C2-066 — Credit Scoring & Underwriting Assessment Report Agent
# Cat 2 RAG (nested pattern): outer backbone (AgentBaseGraph) + inner
# domain workflow (BaseGraph).  Retrieval-grounded underwriting-policy
# assessment.  Fields below cover both layers.
#
# ADR-005 compliance: all dict/list-valued fields are stored as JSON-
# serialized Optional[str].  Use to_json() / from_json() helpers below
# at every producer and consumer node — one contract end-to-end.
# Never type a dict/list field as a bare dict/list; that causes msgpack
# serialization failures and a CoE Stage-6 state-contract finding.
#
# PII note: applicant credit data is individual credit information (個人情報,
# APPI).  No credential/PII field names are stored in State, and nothing is
# persisted beyond the invocation.

import json
from typing import Any, NotRequired, Optional

from framework.schemas.agent_state import AgentState


def to_json(value: Any) -> Optional[str]:
    """Serialize a value to a JSON string for State storage (ADR-005)."""
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def from_json(value: Optional[str], default: Any = None) -> Any:
    """Deserialize a JSON string from State storage (ADR-005)."""
    if value is None:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


class State(AgentState):
    """Flat TypedDict for FIN-C2-066 (Cat 2 RAG, nested).

    All shared fields (user_input, status, session_id, node_history,
    error_log, hitl_*, etc.) are inherited from AgentState.

    ADR-005: dict/list fields use JSON-serialized Optional[str].
    formatted_output is NOT re-declared here — it is inherited from AgentState.
    """

    # ------------------------------------------------------------------
    # Outer layer — set by PreProcessNode (pre_process backbone, S-1)
    # ------------------------------------------------------------------

    # Validated / normalised underwriting query produced by PreProcessNode;
    # consumed by the inner InputValidateNode.
    validated_input: NotRequired[Optional[str]]

    # JSON-serialised channel/request metadata dict (ADR-005: stored as str).
    # Shape: {"source": str, "channel": str}
    enriched_context: NotRequired[Optional[str]]

    # ------------------------------------------------------------------
    # Runtime config — seeded by DomainWorkflowGraph._extra_initial_state()
    # from config/config.yaml (forwarded via the outer _parent_config()),
    # after finite + bounded parsing.
    # Scalars, NOT JSON-serialised: ADR-005 JSON-wrapping applies to
    # dict/list fields only.  RetrieveNode / RerankFilterNode read these
    # (state → config["configurable"] → node default).
    # ------------------------------------------------------------------

    # Number of candidate passages RetrieveNode returns (retrieval.top_k).
    top_k: NotRequired[Optional[int]]

    # Minimum relevance score RerankFilterNode keeps (retrieval.score_threshold).
    score_threshold: NotRequired[Optional[float]]

    # Max passages RerankFilterNode keeps after filtering (retrieval.rerank_keep).
    rerank_keep: NotRequired[Optional[int]]

    # ------------------------------------------------------------------
    # Inner layer — domain nodes (DomainWorkflowGraph)
    # ------------------------------------------------------------------

    # Canonical underwriting question (whitespace-normalised) for retrieval.
    credit_query: NotRequired[Optional[str]]

    # JSON-serialised list of retrieved policy passages (ADR-005: stored as str).
    # Each item: {"id": str, "title": str, "text": str, "score": float}
    retrieved_documents: NotRequired[Optional[str]]

    # JSON-serialised retrieval metadata (ADR-005: stored as str).
    # Shape: {"top_k": int, "candidate_count": int, "retrieved_count": int}
    retrieval_metadata: NotRequired[Optional[str]]

    # JSON-serialised reranked + threshold-filtered passages (ADR-005: stored as str).
    reranked_documents: NotRequired[Optional[str]]

    # Grounded underwriting-assessment answer text (with advisory disclaimer).
    generated_answer: NotRequired[Optional[str]]

    # JSON-serialised list of cited passages (ADR-005: stored as str).
    # Each item: {"id": str, "title": str}
    answer_citations: NotRequired[Optional[str]]

    # True if the answer is grounded in at least one reranked passage.
    grounded: NotRequired[Optional[bool]]

    # Final formatted underwriting assessment (advisory), assembled by the
    # inner OutputFormatNode and S-3-gated by the outer PostProcessNode.
    assessment_report: NotRequired[Optional[str]]

    # ------------------------------------------------------------------
    # Outer layer — set by PostProcessNode (post_process backbone, S-3)
    # ------------------------------------------------------------------

    # Primary result surfaced to the caller (same content as assessment_report
    # after the S-3 gate passes).  formatted_output (from AgentState) is also set.
    result: NotRequired[Optional[str]]

    # ------------------------------------------------------------------
    # Tracing / audit — framework-managed; do NOT write from node code
    # ------------------------------------------------------------------

    trace_id: NotRequired[Optional[str]]
    correlation_id: NotRequired[Optional[str]]
    # node_history inherited from AgentState
