"""AgentCore Platform v1.0"""

# FIN-C2-066 — DomainWorkflowGraph (inner BaseGraph)
#
# This is the INNER graph for the Cat 2 two-layer nested architecture.
# It encapsulates the retrieval-grounded underwriting-assessment pipeline:
#
#   START
#     → input_validate    (InputValidateNode)
#     → retrieve          (RetrieveNode)
#     → rerank_filter     (RerankFilterNode)
#     → generate_answer   (GenerateAnswerNode)   [grounded synthesis]
#     → output_format     (OutputFormatNode)
#     → END
#
# Called by CreditAssessmentGraphNode.get_subgraph() (graph.py).
# get_output() shapes the sub_result dict consumed by merge_output() there.
#
# Rules enforced:
#   ✅ Inherits BaseGraph (fully custom topology — no forced backbone)
#   ✅ Implements all 7 BaseGraph ABC methods
#   ✅ register_nodes() does NOT call super() (abstract in BaseGraph)
#   ✅ Does NOT register initialize / finalize (outer backbone concerns)
#   ✅ All inner nodes declare required_trust_level = TrustLevel.ANONYMOUS
#   ✅ _extra_initial_state() seeds config/config.yaml values, finite and bounded
#   ✅ get_output() designed together with CreditAssessmentGraphNode.merge_output()
#   ✅ All inner node ctors are empty-parens (no constructor args — CoE no-arg rule)
#   ❌ No Level-0 platform SDK imports

from typing import Any, Dict

from langgraph.graph import END, START

from framework.graph.base_graph import BaseGraph
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from src.nodes.generate_answer_node import GenerateAnswerNode
from src.nodes.input_validate_node import InputValidateNode
from src.nodes.output_format_node import OutputFormatNode
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import State
from src.security.screens import finite_float, finite_int

# Upper bounds for the declared retrieval parameters.  They cap work per
# invocation as well as rejecting nonsense: the in-module policy corpus is small,
# and a caller-visible report cites at most a handful of passages.
_MAX_TOP_K = 50
_MAX_RERANK_KEEP = 50


class DomainWorkflowGraph(BaseGraph):
    """Inner domain workflow graph for FIN-C2-066 (Cat 2 RAG).

    Inherits BaseGraph directly for a fully custom node topology.
    Called by CreditAssessmentGraphNode.get_subgraph() in graph.py.

    Pipeline (linear):
        START
          → input_validate    (InputValidateNode)
          → retrieve          (RetrieveNode)
          → rerank_filter     (RerankFilterNode)
          → generate_answer   (GenerateAnswerNode)
          → output_format     (OutputFormatNode)
          → END

    All nodes are FunctionNode subclasses with ANONYMOUS trust_level.
    initialize / finalize are outer backbone concerns — not registered here.
    """

    # ── Identity ──────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "fin_c2_066_credit_underwriting_assessment_workflow"

    @property
    def state_schema(self) -> type:
        return State

    # ── Config validation ─────────────────────────────────────────────────────

    def _validate_config(self) -> None:
        """No mandatory config for the v1 rule-based inner graph."""
        pass

    # ── Runtime config seeding ────────────────────────────────────────────────

    def _extra_initial_state(self) -> Dict[str, Any]:
        """Seed the declared runtime config into the inner initial state.

        The SDK does not thread the graph's ``self.config`` into
        node.execute()'s ``config`` argument on the real ``.invoke()`` path (that
        argument is only populated by direct unit calls).  The outer GraphNode
        forwards config/config.yaml's declared retrieval settings via
        _parent_config() → ``self.config["configurable"]``; here they are copied
        into the inner state so RetrieveNode and RerankFilterNode observe them
        end-to-end.

        Every value is parsed FINITE and BOUNDED.  ``float("NaN")`` parses
        cleanly and then compares False against every threshold, so an
        unvalidated ``score_threshold: NaN`` would drop every retrieved passage
        and return a confident "no policy applies" answer instead of failing —
        the filter disabled in silence.  Out-of-range, non-finite and
        unparseable values are omitted so the nodes fall back to their declared
        defaults.
        """
        configurable = (self.config or {}).get("configurable", {}) or {}
        seeded: Dict[str, Any] = {}
        top_k = finite_int(configurable.get("top_k"), minimum=1, maximum=_MAX_TOP_K)
        if top_k is not None:
            seeded["top_k"] = top_k
        threshold = finite_float(configurable.get("score_threshold"), minimum=0.0, maximum=1.0)
        if threshold is not None:
            seeded["score_threshold"] = threshold
        rerank_keep = finite_int(configurable.get("rerank_keep"), minimum=1, maximum=_MAX_RERANK_KEEP)
        if rerank_keep is not None:
            seeded["rerank_keep"] = rerank_keep
        return seeded

    # ── Node registration ─────────────────────────────────────────────────────

    def register_nodes(self) -> None:
        """Register all 5 domain nodes.

        No super() call — BaseGraph.register_nodes() is abstract.
        Do NOT register initialize or finalize; those are outer backbone
        concerns handled by AgentBaseGraph in graph.py.  Every key registered
        here is referenced in add_edges().  All nodes are instantiated with
        empty-parens (no ctor args) — CoE no-arg ctor rule.
        """
        self._nodes["input_validate"] = InputValidateNode()
        self._nodes["retrieve"] = RetrieveNode()
        self._nodes["rerank_filter"] = RerankFilterNode()
        self._nodes["generate_answer"] = GenerateAnswerNode()
        self._nodes["output_format"] = OutputFormatNode()

    # ── Edge wiring ───────────────────────────────────────────────────────────

    def add_edges(self) -> None:
        """Wire the linear retrieval-grounded assessment topology."""
        self._sg.add_edge(START, "input_validate")
        self._sg.add_edge("input_validate", "retrieve")
        self._sg.add_edge("retrieve", "rerank_filter")
        self._sg.add_edge("rerank_filter", "generate_answer")
        self._sg.add_edge("generate_answer", "output_format")
        self._sg.add_edge("output_format", END)

    # ── Routing ───────────────────────────────────────────────────────────────

    def route(self, state: AgentState) -> str:
        """Conditional routing — required by BaseGraph ABC.

        Linear topology; add_conditional_edges() is not used, so this method is
        never called at runtime.  Returns END on error so an unexpected
        invocation does not re-enter a processing node.
        """
        if state.get("status") == AgentStatus.ERROR.value:
            return END
        return "output_format"

    # ── Output shape ──────────────────────────────────────────────────────────

    def get_output(self, state: AgentState) -> Dict[str, Any]:
        """Shape the output dict returned to the outer graph as sub_result.

        Received by CreditAssessmentGraphNode.merge_output() in graph.py.  Both
        methods are designed together to guarantee field-name consistency.
        """
        return {
            "assessment_report": state.get("assessment_report"),
            "generated_answer": state.get("generated_answer"),
            "answer_citations": state.get("answer_citations"),
            "grounded": state.get("grounded", False),
            "result": state.get("result"),
            "status": state.get("status"),
            "node_history": state.get("node_history", []),
            "correlation_id": state.get("correlation_id"),
        }
