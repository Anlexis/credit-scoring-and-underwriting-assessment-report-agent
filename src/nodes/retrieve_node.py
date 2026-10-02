"""AgentCore Platform v1.0"""

# FIN-C2-066 — RetrieveNode
# Inner RAG domain node 2: retrieve the top_k candidate underwriting-policy
# passages for the credit_query.
#
# v1: deterministic keyword-overlap retrieval over a small in-module corpus of
# underwriting-policy passages.  Production wires the real vector store /
# embedding retriever; the node contract (credit_query -> retrieved_documents)
# is unchanged.
#
# Runtime config: top_k comes from config/config.yaml (retrieval.top_k),
# forwarded by the outer _parent_config() and seeded — finite and bounded — into
# inner state by DomainWorkflowGraph._extra_initial_state().
# Read precedence: state["top_k"] (real .invoke() path) -> config["configurable"]
# (direct unit calls) -> _DEFAULT_TOP_K.
#
# Inner node — ANONYMOUS trust (review finding 5, corrected 2026-07-02).
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
from typing import Any, ClassVar, Dict, List, Optional, Set

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import to_json

logger = logging.getLogger(__name__)

_DEFAULT_TOP_K = 5

# In-module underwriting-policy knowledge base (v1).  Each passage is a small,
# self-contained policy statement.  Production replaces this with a vector store.
_KB_CORPUS: List[Dict[str, str]] = [
    {
        "id": "UW-DTI-01",
        "title": "Debt-to-Income (DTI) threshold",
        "text": (
            "Standard underwriting caps the total debt-to-income (DTI) ratio at 35% for "
            "unsecured consumer credit and 43% for mortgage-backed applications. "
            "Applications above the cap require a compensating-factor review and manager sign-off."
        ),
    },
    {
        "id": "UW-SCORE-02",
        "title": "Credit score bands",
        "text": (
            "Internal credit-score bands: A (>=750) auto-eligible, B (700-749) standard, "
            "C (640-699) enhanced review, D (<640) decline unless secured by collateral. "
            "Scores are advisory inputs, not the sole decision basis."
        ),
    },
    {
        "id": "UW-COLL-03",
        "title": "Collateral and loan-to-value",
        "text": (
            "Secured products require a loan-to-value (LTV) ratio of 80% or lower. "
            "Collateral valuations must be dated within 90 days and sourced from an approved appraiser."
        ),
    },
    {
        "id": "UW-INCOME-04",
        "title": "Income verification",
        "text": (
            "Income must be verified via two consecutive payslips or the latest tax return. "
            "Self-employed applicants provide two years of filed returns; irregular income is "
            "averaged over 24 months."
        ),
    },
    {
        "id": "UW-DELINQ-05",
        "title": "Delinquency and adverse history",
        "text": (
            "Any charge-off or 90+ day delinquency within 24 months triggers a manual "
            "adverse-history review. A single 30-day late over 12 months old is not, by "
            "itself, a decline reason."
        ),
    },
    {
        "id": "UW-REG-06",
        "title": "Regulatory documentation (FSA / APPI)",
        "text": (
            "Under FSA AI guidance, AI-assisted credit decisions must retain a documented "
            "rationale and an audit trail. Applicant personal data (individual credit "
            "information) is processed under APPI and must not be retained beyond the assessment."
        ),
    },
]

_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "of",
        "to",
        "for",
        "and",
        "or",
        "in",
        "on",
        "what",
        "which",
        "how",
        "do",
        "does",
        "my",
        "i",
        "with",
        "at",
        "be",
        "can",
    }
)


def _tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokenisation."""
    return [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t]


def _score(query_tokens: Set[str], passage: Dict[str, str]) -> float:
    """Keyword-overlap relevance score in [0, 1]."""
    doc_tokens = set(_tokenize(passage["title"] + " " + passage["text"])) - _STOPWORDS
    q = query_tokens - _STOPWORDS
    if not q or not doc_tokens:
        return 0.0
    overlap = len(q & doc_tokens)
    return round(overlap / len(q), 4)


class RetrieveNode(FunctionNode):
    """Retrieve top_k candidate underwriting-policy passages for the credit_query.

    Inner node — ANONYMOUS trust (see module docstring).

    Input state keys:
        credit_query: str  — canonical query from InputValidateNode
        top_k:        int  — seeded from config/config.yaml (optional)

    Output state keys (partial dict):
        retrieved_documents: str  — JSON list of {id,title,text,score} (ADR-005)
        retrieval_metadata:  str  — JSON metadata dict (ADR-005)
        status:              str
        error_log:           list[str]  (only on ERROR)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        query = state.get("credit_query") or state.get("validated_input") or ""
        if not isinstance(query, str) or not query.strip():
            emit_trace_event(
                "retrieve_failed",
                {"reason": "missing_query"},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["RetrieveNode: credit_query missing in state"],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. "
                + ("RetrieveNode: credit_query missing in state"),
            }

        # top_k precedence: declared value seeded into inner state by
        # DomainWorkflowGraph._extra_initial_state() (the real .invoke() path) ->
        # config["configurable"] (direct unit calls) -> default.
        configurable = (config or {}).get("configurable", {})
        top_k_val = state.get("top_k")
        if top_k_val is None:
            top_k_val = configurable.get("top_k", _DEFAULT_TOP_K)
        top_k = int(top_k_val)

        query_tokens = set(_tokenize(query))
        scored: List[Dict[str, Any]] = []
        for passage in _KB_CORPUS:
            score = _score(query_tokens, passage)
            if score > 0.0:
                scored.append(
                    {
                        "id": passage["id"],
                        "title": passage["title"],
                        "text": passage["text"],
                        "score": score,
                    }
                )
        scored.sort(key=lambda d: d["score"], reverse=True)
        retrieved = scored[:top_k]

        logger.info(
            "RetrieveNode: candidates=%d retrieved=%d top_k=%d",
            len(scored),
            len(retrieved),
            top_k,
        )
        emit_trace_event(
            "retrieve_complete",
            {
                "candidate_count": len(scored),
                "retrieved_count": len(retrieved),
                "top_k": top_k,
            },
            state,
        )

        return {
            "retrieved_documents": to_json(retrieved),
            "retrieval_metadata": to_json(
                {
                    "top_k": top_k,
                    "candidate_count": len(scored),
                    "retrieved_count": len(retrieved),
                }
            ),
            "status": AgentStatus.SUCCESS.value,
        }
