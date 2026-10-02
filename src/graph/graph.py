"""AgentCore Platform v1.0"""

# FIN-C2-066 — Outer graph (AgentBaseGraph; Cat 2 two-layer nested architecture)
#
# Architecture (Cat 2):
#
#   Outer backbone (fixed — identical to Cat 1, do NOT override add_edges()):
#     START → initialize → pre_process → main → {route} → post_process → finalize → END
#                                             ↓ (RETRY, max 3)
#                                          pre_process
#
#   The `main` slot is a GraphNode subclass (CreditAssessmentGraphNode) that
#   delegates the full domain workflow to DomainWorkflowGraph (inner BaseGraph).
#
# Directory layout:
#   src/graph/graph.py                 ← outer graph (this file)
#   src/graph/domain_workflow_graph.py ← inner graph (RAG topology)
#
# Rules enforced:
#   ✅ CreditScoringUnderwritingAssessmentReportAgent inherits AgentBaseGraph (L1 Base)
#   ✅ super().register_nodes() called first (fills initialize + finalize)
#   ✅ CreditAssessmentGraphNode assigned to self._nodes["main"]
#   ✅ PreProcessNode (VERIFIED_EXTERNAL) in pre_process slot (S-1 gate)
#   ✅ PostProcessNode (ANONYMOUS) in post_process slot (S-3 output gate)
#   ✅ S-3 _security_gate_output declared on the agent class (the framework rules §5-6)
#   ✅ merge_output() returns only changed keys
#   ✅ _parent_config() forwards config/config.yaml runtime settings
#   ✅ get_output() surfaces the gated domain result on SUCCESS, and only there
#   ✅ class name matches config/agent.yaml class: field exactly
#   ❌ add_edges() NOT overridden on the outer graph
#   ❌ No Level-0 platform SDK imports

import os
from typing import Any, ClassVar, Dict, Optional

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.nodes.graph_node import GraphNode
from framework.schemas.trust_level import TrustLevel
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import State
from src.security.screens import detect_output_credentials

# Runtime parameters — config/config.yaml at the repo root (three levels up from
# this file: src/graph/graph.py → src/graph → src → <repo root>).
#
# NOT config/agent.yaml.  The manifest is flat — AgentRegistry reads every key at
# root level and there is no `agent:` block — so the reader that used to walk
# `manifest["agent"]["config"]` now returns nothing and every declared value goes
# dead.  It degrades invisibly, because the node defaults happen to equal the
# declared values; the config-liveness test drives a changed value end-to-end so
# a regression here fails instead of going quiet.
_RUNTIME_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config",
    "config.yaml",
)


def _load_runtime_config() -> Dict[str, Any]:
    """Return the parsed contents of config/config.yaml.

    Best-effort: a missing or unparseable file yields ``{}`` so graph
    construction never breaks (the inner nodes then fall back to their declared
    defaults). PyYAML is loaded lazily — it is a framework runtime dependency,
    so importing it on demand avoids a hard module-load coupling.
    """
    try:
        import yaml

        with open(_RUNTIME_CONFIG_PATH, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


class CreditAssessmentGraphNode(GraphNode):
    """GraphNode subclass assigned to the `main` slot of the agent.

    Wraps DomainWorkflowGraph (inner Cat 2 BaseGraph).  Called by the
    AgentBaseGraph backbone after pre_process and before post_process.

    Contracts:
      get_subgraph()  — instantiate and return DomainWorkflowGraph
      extract_input() — pull validated_input (S-1 output) from outer state
      merge_output()  — map sub_result fields into outer state delta (changed keys only)
      error_strategy  — "propagate": re-raise inner errors as SubgraphError (fail-fast)
    """

    # S-1 declared on the wrapper too: the CI gate only AST-scans FunctionNode
    # subclasses, so a GraphNode main slot passes the pipeline without one and is
    # flagged at review. Same level the nodes in this repo already declare.
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    error_strategy: ClassVar[str] = "propagate"
    propagate_hitl: ClassVar[bool] = False

    def get_subgraph(self) -> Any:
        """Instantiate and return the inner domain workflow graph.

        DomainWorkflowGraph is imported lazily (inside the method) to avoid
        circular-import risk at module load time and to match the Cat 2 pattern.
        """
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        return DomainWorkflowGraph(config=self._parent_config())

    def extract_input(self, state: AgentState) -> str:
        """Return the string input passed into inner_graph.invoke().

        PreProcessNode (S-1) validates and normalises the raw user_input and
        writes the result to validated_input.  Prefer that; fall back to
        user_input if validated_input is absent (e.g. in unit tests).
        """
        value = state.get("validated_input", state.get("user_input", ""))
        return value if isinstance(value, str) else ""

    def merge_output(self, state: AgentState, sub_result: Dict[str, Any]) -> Dict[str, Any]:
        """Map the inner graph sub_result back into the outer state delta.

        sub_result is the dict returned by DomainWorkflowGraph.get_output().
        Returns ONLY changed keys — never the full state.

        Key coupling (designed together with DomainWorkflowGraph.get_output()):
          Inner get_output() emits  → "assessment_report", "generated_answer",
                                       "answer_citations", "grounded", "result", "status"
          This merge_output() reads → sub_result.get(...) for each of these keys.

        PostProcessNode (outer post_process) reads assessment_report from state
        to apply the S-3 gate and set formatted_output.
        """
        return {
            "assessment_report": sub_result.get("assessment_report"),
            "generated_answer": sub_result.get("generated_answer"),
            "answer_citations": sub_result.get("answer_citations"),
            "grounded": sub_result.get("grounded", False),
            "result": sub_result.get("result"),
            "status": sub_result.get("status"),
        }

    def _parent_config(self) -> Dict[str, Any]:
        """Forward the declared runtime config to the inner graph.

        Reads config/config.yaml (the runtime parameters) and exposes the
        declared settings under the LangGraph ``configurable`` key.
        DomainWorkflowGraph._extra_initial_state() seeds these into the inner
        state so RetrieveNode and RerankFilterNode observe the configured
        top_k / score_threshold / rerank_keep on the real ``.invoke()`` path
        (the SDK does not thread runtime config into node.execute()).  Only keys
        the file actually declares are forwarded; absent keys fall back to node
        defaults.
        """
        cfg = _load_runtime_config()
        retrieval = cfg.get("retrieval", {}) or {}
        llm = cfg.get("llm", {}) or {}
        declared = {
            "top_k": retrieval.get("top_k"),
            "score_threshold": retrieval.get("score_threshold"),
            "rerank_keep": retrieval.get("rerank_keep"),
            "system_prompt_template": llm.get("system_prompt_template"),
            "temperature": llm.get("temperature"),
            "max_tokens": llm.get("max_tokens"),
        }
        return {"configurable": {k: v for k, v in declared.items() if v is not None}}


class CreditScoringUnderwritingAssessmentReportAgent(AgentBaseGraph):
    """Outer graph for FIN-C2-066 (Cat 2 — RAGAgent).

    Inherits AgentBaseGraph directly (L1 Base).  Domain logic is fully
    encapsulated in CreditAssessmentGraphNode (main slot), which delegates to
    DomainWorkflowGraph (inner BaseGraph).

    Backbone (fixed — identical to Cat 1):
        START → initialize → pre_process → main → post_process → finalize → END

    register_nodes() and get_output() are the overrides:
      - super().register_nodes() fills: initialize, finalize (framework defaults)
      - pre_process: PreProcessNode          (VERIFIED_EXTERNAL — S-1 trust gate)
      - main:        CreditAssessmentGraphNode (delegates to DomainWorkflowGraph)
      - post_process: PostProcessNode         (ANONYMOUS — S-3 output gate)
      - get_output(): surfaces the S-3-gated domain result (SR re-review 2026-07)

    add_edges() is NOT overridden — backbone wiring belongs to the framework.

    Class name MUST match config/agent.yaml `class:` field exactly.
    server.py imports this as `Graph` via the alias below.
    """

    # Mandatory output gate (the framework rules §5-6): declared on the agent class as the
    # canonical S-3 policy for this template.  Runtime enforcement in
    # PostProcessNode calls the SAME module-level screen, so the declaration and
    # the enforcement cannot drift apart — a local pattern set that fell behind
    # the framework's would be a bypass, because the framework's own @final gate
    # then raises inside the node and the wrapper discards the node's clearing.
    # Runtime enforcement stays module-level (a node instance method would be
    # auto-wrapped by the real SDK → None-state AttributeError; CoE findings
    # A peer template / a peer template).

    def _security_gate_output(self, content: str) -> Optional[str]:
        """Mandatory output gate for the agent (the framework rules §5-6).

        Return the class name of the first credential/secret pattern found in
        the assessment output, or None if the output is clean.  PostProcessNode
        applies the same screen at runtime.
        """
        return detect_output_credentials(content or "")

    @property
    def name(self) -> str:
        """Agent identifier registered with AgentRegistry."""
        return "CreditScoringUnderwritingAssessmentReportAgent"

    @property
    def state_schema(self) -> type:
        return State

    def register_nodes(self) -> None:
        """Fill all 5 backbone slots.

        super().register_nodes() MUST be called first — it injects the
        framework's default InitializeNode (sets schema_version, session_id,
        trust_level) and FinalizeNode (builds response_metadata, total_time_ms).
        """
        super().register_nodes()  # fills: initialize, finalize

        self._nodes["pre_process"] = PreProcessNode()
        self._nodes["main"] = CreditAssessmentGraphNode()
        self._nodes["post_process"] = PostProcessNode()

    def get_output(self, state: AgentState) -> Dict[str, Any]:
        """Surface the gated underwriting assessment on the outer invoke() return.

        AgentBaseGraph.get_output() returns only the minimal
        ``{output, status, trace_id, correlation_id, node_history}`` envelope, so
        on the success path the structured domain result that
        CreditAssessmentGraphNode.merge_output() merges into outer state
        (generated_answer, answer_citations, grounded, assessment_report, result)
        never reached the caller.  This override adds it back — on the success
        path only.

        **Containment.** The base envelope's ``output`` is
        ``formatted_output or result``, with no status check, and
        ``BaseNode.__call__`` turns an exception inside post_process into a bare
        ERROR partial that clears nothing.  So if PostProcessNode crashes or
        never runs, outer state still holds the PRE-GATE assessment that
        merge_output() wrote, and an unconditional ``output["result"] =
        state.get("result")`` hands the caller exactly the text the gate was
        there to withhold.  Measured on this repo before the fix: a data-path
        fault in merge_output produced ``status=error`` carrying the complete
        un-gated report in both ``output`` and ``result``.

        Therefore:

        * SUCCESS is the only path that surfaces content, and reaching SUCCESS
          means PostProcessNode ran and its gate passed — it is the only node
          that sets SUCCESS at post_process, and the only writer of
          ``formatted_output``.
        * On ANY non-success outcome the envelope is re-resolved from
          ``formatted_output`` alone — the gate's own truthy notice, or None.
          ``result`` is never surfaced from state there, and every structured
          field is withheld.
        """
        # dict() so the annotated local is a real mapping, not the wheel's Any.
        output: Dict[str, Any] = dict(super().get_output(state))
        succeeded = state.get("status") == AgentStatus.SUCCESS.value

        if not succeeded:
            notice = state.get("formatted_output")
            withheld = notice if isinstance(notice, str) and notice else None
            output["output"] = withheld
            output["formatted_output"] = withheld
            output["result"] = withheld
            output["assessment_report"] = None
            output["generated_answer"] = None
            output["answer_citations"] = None
            output["grounded"] = None
            return output

        gated_report = state.get("formatted_output") or state.get("result")
        output["formatted_output"] = state.get("formatted_output")
        output["result"] = state.get("result")
        output["assessment_report"] = gated_report
        output["generated_answer"] = state.get("generated_answer")
        output["answer_citations"] = state.get("answer_citations")
        output["grounded"] = state.get("grounded")
        return output

    # add_edges() is NOT overridden — backbone wiring belongs to the framework.


# Alias for backward compat (server.py imports Graph).
# Class name CreditScoringUnderwritingAssessmentReportAgent matches config/agent.yaml class: field.
Graph = CreditScoringUnderwritingAssessmentReportAgent
