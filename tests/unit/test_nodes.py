# FIN-C2-066 — Unit Tests: domain nodes + graph wiring
#
# Real, non-stub unit tests. They import the REAL modules merged to develop in
# Wave-1 and assert real behaviour (RAG grounding + abstention, credit-underwriting
# assessment content, S-1 trust levels, the S-3 output gate, and the Cat 2
# two-layer nested graph composition).
#
# S-4 audit events are patched at the node MODULE level (not via a sys.modules
# stub, which would break the real `shared` package the framework loads at import
# time). Patch pattern per node:
#     monkeypatch.setattr("src.nodes.<mod>.emit_trace_event", lambda *a, **k: None)

import json

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from src.schemas.state import from_json, to_json


# ── Shared helpers ────────────────────────────────────────────────────────────

# An underwriting query that overlaps several KB passages (DTI, credit-score
# bands, income verification) so retrieval is grounded.
GROUNDED_QUERY = (
    "What debt-to-income (DTI) ratio and credit-score band thresholds apply "
    "when underwriting an unsecured consumer credit application, and what "
    "income verification is required?"
)
GROUNDED_PAYLOAD = json.dumps({"query": GROUNDED_QUERY})

# An off-topic query with no KB overlap -> retrieval empty -> abstention path.
OFFTOPIC_QUERY = "zzz qqq wibble frobnicate"


# ── PreProcessNode (outer pre_process, S-1 VERIFIED_EXTERNAL) ──────────────────


class TestPreProcessNode:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.pre_process_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.pre_process_node import PreProcessNode

        self.node = PreProcessNode()

    def test_plain_string_query_returns_success(self):
        result = self.node(
            {
                "user_input": GROUNDED_QUERY,
                "input_context": {},
                "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
            }
        )
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["validated_input"] == GROUNDED_QUERY

    def test_json_object_query_is_extracted(self):
        result = self.node(
            {
                "user_input": GROUNDED_PAYLOAD,
                "input_context": {},
                "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
            }
        )
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["validated_input"] == GROUNDED_QUERY

    def test_enriched_context_carries_channel(self):
        result = self.node(
            {
                "user_input": GROUNDED_QUERY,
                "input_context": {"channel": "risk-desk"},
                "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
            }
        )
        ctx = from_json(result["enriched_context"])
        assert ctx["channel"] == "risk-desk"
        assert ctx["source"] == "CreditScoringUnderwritingAssessmentReportAgent"

    def test_empty_input_returns_error(self):
        result = self.node(
            {"user_input": "", "input_context": {}, "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value}
        )
        assert result["status"] == AgentStatus.ERROR.value
        assert any("empty" in e for e in result["error_log"])

    def test_json_object_without_query_key_returns_error(self):
        result = self.node(
            {
                "user_input": json.dumps({"foo": "bar"}),
                "input_context": {},
                "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
            }
        )
        assert result["status"] == AgentStatus.ERROR.value
        assert any("query" in e for e in result["error_log"])

    def test_overlong_query_is_rejected(self):
        # SR Medium fix (2026-07): overlength input is rejected with a validation
        # error, NOT silently truncated (which would alter an assessment request
        # without informing the caller).
        long_q = "credit " * 1000  # > 4000 chars
        result = self.node(
            {"user_input": long_q, "input_context": {}, "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value}
        )
        assert result["status"] == AgentStatus.ERROR.value
        assert any("too long" in e for e in result["error_log"])

    def test_trust_level_is_verified_external(self):
        assert self.node.required_trust_level == TrustLevel.VERIFIED_EXTERNAL

    def test_execute_signature_is_state_first(self):
        """Every concrete node, not just this one — discovered, not listed.

        A node added later is covered without anyone remembering to add a row.
        """
        import importlib
        import inspect
        import pkgutil

        from framework.nodes.base_node import BaseNode

        pkg = importlib.import_module("src.nodes")
        discovered = []
        for _, modname, _ in pkgutil.walk_packages(pkg.__path__, prefix="src.nodes."):
            module = importlib.import_module(modname)
            for attr in vars(module).values():
                if (
                    isinstance(attr, type)
                    and issubclass(attr, BaseNode)
                    and attr is not BaseNode
                    and attr.__module__ == modname
                    and not inspect.isabstract(attr)
                ):
                    discovered.append(attr)

        assert discovered, "no concrete nodes discovered under src/nodes/"
        for node_cls in discovered:
            params = list(inspect.signature(node_cls.execute).parameters.keys())
            assert params[0] == "self" and params[1] == "state", node_cls.__name__
            assert "_invoke_impl" not in node_cls.__dict__, node_cls.__name__
            assert "process" not in node_cls.__dict__, node_cls.__name__


# ── S-1 trust gate (BaseNode.__call__, PreProcessNode) ─────────────────────────


class TestS1TrustGate:
    """CoE Stage-6 R1 finding: unit tests must route node invocations through
    BaseNode.__call__ so the S-1 trust gate is exercised, not bypassed by a
    direct .execute() call.

    PreProcessNode carries required_trust_level = VERIFIED_EXTERNAL (the only
    outer node that enforces trust). __call__ runs the S-1 gate BEFORE execute():
    an ANONYMOUS caller is denied (never reaches execute()); a VERIFIED_EXTERNAL
    caller is admitted and runs execute() to SUCCESS.
    """

    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.pre_process_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.pre_process_node import PreProcessNode

        self.node = PreProcessNode()

    def test_anonymous_caller_denied_by_s1_gate(self):
        # ANONYMOUS < VERIFIED_EXTERNAL -> __call__ denies before execute().
        result = self.node(
            {
                "user_input": GROUNDED_QUERY,
                "input_context": {},
                "caller_trust_level": TrustLevel.ANONYMOUS.value,
            }
        )
        assert result["status"] == AgentStatus.ERROR.value
        assert any("trust gate denied" in e.lower() for e in result.get("error_log", []))
        # execute() never ran, so its output key is absent (gate short-circuits).
        assert "validated_input" not in result

    def test_verified_external_caller_admitted(self):
        result = self.node(
            {
                "user_input": GROUNDED_QUERY,
                "input_context": {},
                "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
            }
        )
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["validated_input"] == GROUNDED_QUERY


# ── InputValidateNode (inner domain node 1, ANONYMOUS) ─────────────────────────


class TestInputValidateNode:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.input_validate_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.input_validate_node import InputValidateNode

        self.node = InputValidateNode()

    def test_valid_input_builds_credit_query(self):
        result = self.node({"validated_input": GROUNDED_QUERY, "caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["credit_query"] == GROUNDED_QUERY

    def test_whitespace_is_normalised(self):
        result = self.node(
            {"validated_input": "  DTI    threshold\n\tfor credit  ", "caller_trust_level": TrustLevel.ANONYMOUS.value}
        )
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["credit_query"] == "DTI threshold for credit"

    def test_falls_back_to_user_input(self):
        result = self.node({"user_input": GROUNDED_QUERY, "caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["credit_query"] == GROUNDED_QUERY

    def test_empty_query_returns_error(self):
        result = self.node({"validated_input": "", "caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["status"] == AgentStatus.ERROR.value
        assert any("empty" in e for e in result["error_log"])

    def test_too_short_query_returns_error(self):
        result = self.node({"validated_input": "ab", "caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["status"] == AgentStatus.ERROR.value
        assert any("short" in e for e in result["error_log"])

    def test_overlong_query_is_rejected(self):
        # SR Medium fix (2026-07): overlength is rejected, not truncated
        # (consistent with PreProcessNode).
        result = self.node({"validated_input": "credit " * 1000, "caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["status"] == AgentStatus.ERROR.value
        assert any("too long" in e for e in result["error_log"])

    def test_trust_level_anonymous(self):
        assert self.node.required_trust_level == TrustLevel.ANONYMOUS


# ── RetrieveNode (inner domain node 2, ANONYMOUS) ──────────────────────────────


class TestRetrieveNode:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.retrieve_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.retrieve_node import RetrieveNode

        self.node = RetrieveNode()

    def test_grounded_query_retrieves_passages(self):
        result = self.node({"credit_query": GROUNDED_QUERY, "caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["status"] == AgentStatus.SUCCESS.value
        docs = from_json(result["retrieved_documents"])
        assert len(docs) > 0
        ids = {d["id"] for d in docs}
        assert "UW-DTI-01" in ids
        meta = from_json(result["retrieval_metadata"])
        assert meta["retrieved_count"] == len(docs)

    def test_offtopic_query_retrieves_nothing(self):
        # No KB overlap is a valid no-answer (abstention) state — not an error.
        result = self.node({"credit_query": OFFTOPIC_QUERY, "caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["status"] == AgentStatus.SUCCESS.value
        docs = from_json(result["retrieved_documents"])
        assert docs == []
        meta = from_json(result["retrieval_metadata"])
        assert meta["candidate_count"] == 0

    def test_top_k_is_respected(self):
        result = self.node.execute({"credit_query": GROUNDED_QUERY}, {"configurable": {"top_k": 2}})
        docs = from_json(result["retrieved_documents"])
        assert len(docs) <= 2

    def test_top_k_from_seeded_state_is_respected(self):
        # SR Medium fix (2026-07): the manifest-seeded state value (real .invoke()
        # path) takes precedence and is honoured by retrieval.
        result = self.node(
            {"credit_query": GROUNDED_QUERY, "top_k": 1, "caller_trust_level": TrustLevel.ANONYMOUS.value}
        )
        docs = from_json(result["retrieved_documents"])
        assert len(docs) <= 1

    def test_missing_query_returns_error(self):
        result = self.node({"caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["status"] == AgentStatus.ERROR.value
        assert any("credit_query" in e for e in result["error_log"])

    def test_trust_level_anonymous(self):
        assert self.node.required_trust_level == TrustLevel.ANONYMOUS


# ── RerankFilterNode (inner domain node 3, ANONYMOUS) ──────────────────────────


class TestRerankFilterNode:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.rerank_filter_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.rerank_filter_node import RerankFilterNode

        self.node = RerankFilterNode()

    def _docs(self):
        return [
            {"id": "UW-DTI-01", "title": "DTI", "text": "...", "score": 0.5},
            {"id": "UW-SCORE-02", "title": "Score", "text": "...", "score": 0.2},
            {"id": "UW-COLL-03", "title": "Collateral", "text": "...", "score": 0.05},
        ]

    def test_reranks_and_filters_by_threshold(self):
        result = self.node(
            {"retrieved_documents": to_json(self._docs()), "caller_trust_level": TrustLevel.ANONYMOUS.value}
        )
        assert result["status"] == AgentStatus.SUCCESS.value
        kept = from_json(result["reranked_documents"])
        # 0.05 < default 0.1 threshold -> dropped; two remain, sorted by score.
        assert [d["id"] for d in kept] == ["UW-DTI-01", "UW-SCORE-02"]

    def test_keep_limit_is_respected(self):
        docs = [{"id": f"D{i}", "title": "t", "text": "x", "score": 0.9} for i in range(6)]
        result = self.node.execute({"retrieved_documents": to_json(docs)}, {"configurable": {"rerank_keep": 3}})
        kept = from_json(result["reranked_documents"])
        assert len(kept) == 3

    def test_seeded_state_threshold_and_keep_are_respected(self):
        # SR Medium fix (2026-07): manifest-seeded state values (real .invoke()
        # path) take precedence over the node defaults.
        docs = [{"id": f"D{i}", "title": "t", "text": "x", "score": 0.9} for i in range(6)]
        result = self.node(
            {
                "retrieved_documents": to_json(docs),
                "score_threshold": 0.95,
                "rerank_keep": 2,
                "caller_trust_level": TrustLevel.ANONYMOUS.value,
            }
        )
        kept = from_json(result["reranked_documents"])
        # threshold 0.95 filters everything at 0.9 -> empty.
        assert kept == []

    def test_empty_retrieval_yields_empty_reranked(self):
        # Empty retrieval is a valid no-answer state (abstention), not an error.
        result = self.node({"retrieved_documents": to_json([]), "caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert from_json(result["reranked_documents"]) == []

    def test_trust_level_anonymous(self):
        assert self.node.required_trust_level == TrustLevel.ANONYMOUS


# ── GenerateAnswerNode (inner domain node 4, grounded synthesis, ANONYMOUS) ────


class TestGenerateAnswerNode:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.generate_answer_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.generate_answer_node import GenerateAnswerNode

        self.node = GenerateAnswerNode()

    def _passages(self):
        return [
            {"id": "UW-DTI-01", "title": "DTI threshold", "text": "DTI capped at 35%.", "score": 0.5},
        ]

    def test_grounded_answer_cites_passages(self):
        state = {
            "credit_query": GROUNDED_QUERY,
            "reranked_documents": to_json(self._passages()),
            "caller_trust_level": TrustLevel.ANONYMOUS.value,
        }
        result = self.node(state)
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["grounded"] is True
        assert "Advisory only" in result["generated_answer"]
        cites = from_json(result["answer_citations"])
        assert cites[0]["id"] == "UW-DTI-01"

    def test_abstains_when_no_grounding(self):
        # No reranked passages -> abstention (no-answer) path, still SUCCESS.
        state = {
            "credit_query": GROUNDED_QUERY,
            "reranked_documents": to_json([]),
            "caller_trust_level": TrustLevel.ANONYMOUS.value,
        }
        result = self.node(state)
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["grounded"] is False
        assert "no grounded assessment" in result["generated_answer"].lower()
        assert from_json(result["answer_citations"]) == []

    def test_answer_carries_advisory_disclaimer(self):
        state = {
            "credit_query": GROUNDED_QUERY,
            "reranked_documents": to_json(self._passages()),
            "caller_trust_level": TrustLevel.ANONYMOUS.value,
        }
        result = self.node(state)
        assert "does not constitute a credit decision" in result["generated_answer"]

    def test_trust_level_anonymous(self):
        assert self.node.required_trust_level == TrustLevel.ANONYMOUS


# ── OutputFormatNode (inner domain node 5, ANONYMOUS) ──────────────────────────


class TestOutputFormatNode:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.output_format_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.output_format_node import OutputFormatNode

        self.node = OutputFormatNode()

    def _state(self, grounded=True):
        return {
            "generated_answer": "Underwriting assessment body text.",
            "answer_citations": to_json([{"id": "UW-DTI-01", "title": "DTI threshold"}]),
            "grounded": grounded,
            "caller_trust_level": TrustLevel.ANONYMOUS.value,
        }

    def test_assembles_full_report(self):
        result = self.node(self._state())
        assert result["status"] == AgentStatus.SUCCESS.value
        report = result["assessment_report"]
        assert result["result"] == report
        assert "CREDIT UNDERWRITING ASSESSMENT (ADVISORY)" in report
        assert "CITED UNDERWRITING-POLICY PASSAGES" in report
        assert "UW-DTI-01" in report

    def test_grounded_flag_reflected_in_report(self):
        assert "Grounded in retrieved policy: YES" in self.node(self._state(True))["assessment_report"]
        assert "Grounded in retrieved policy: NO" in self.node(self._state(False))["assessment_report"]

    def test_no_citations_note(self):
        state = {
            "generated_answer": "text",
            "answer_citations": to_json([]),
            "grounded": False,
            "caller_trust_level": TrustLevel.ANONYMOUS.value,
        }
        report = self.node(state)["assessment_report"]
        assert "none — no passage met the relevance threshold" in report

    def test_missing_answer_returns_error(self):
        result = self.node(
            {"generated_answer": "", "answer_citations": to_json([]), "caller_trust_level": TrustLevel.ANONYMOUS.value}
        )
        assert result["status"] == AgentStatus.ERROR.value
        assert any("generated_answer" in e for e in result["error_log"])

    def test_trust_level_anonymous(self):
        assert self.node.required_trust_level == TrustLevel.ANONYMOUS


# ── PostProcessNode (outer post_process, S-3 output gate, ANONYMOUS) ───────────


class TestPostProcessNode:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.post_process_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.post_process_node import PostProcessNode

        self.node = PostProcessNode()

    def test_clean_report_passes_gate(self):
        report = "CREDIT UNDERWRITING ASSESSMENT (ADVISORY)\nDTI capped at 35%."
        result = self.node({"assessment_report": report, "caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["formatted_output"] == report
        assert result["result"] == report

    def test_empty_report_uses_fallback(self):
        result = self.node({"assessment_report": "", "caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "No assessment content generated" in result["formatted_output"]

    def test_output_gate_withholds_credential_leak(self):
        """Behaviour, not wording: nothing of the refused assessment survives.

        The notice must be TRUTHY — a falsy formatted_output re-opens the
        framework's `formatted_output or result` fallback, which is the leak
        this branch exists to prevent.
        """
        secret = "sk-abcdefghij0123456789ABCDEF"
        leaky = f"CREDIT UNDERWRITING ASSESSMENT\nDTI capped at 35%.\ntoken={secret}"
        result = self.node({"assessment_report": leaky, "caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["status"] == AgentStatus.ERROR.value
        notice = result["formatted_output"]
        assert notice, "the replacement must be truthy or the base fallback re-opens"
        assert result["result"] == notice
        # Neither the secret nor any released text of the refused assessment.
        for field in ("formatted_output", "result"):
            assert secret not in result[field]
            assert "DTI capped at 35%" not in result[field]
        assert any("output gate refused" in e for e in result["error_log"])
        assert secret not in " ".join(result["error_log"])

    def test_output_gate_clears_every_output_bearing_field(self):
        """Presence AND emptiness — omitting a key leaves the OLD value in state.

        LangGraph merges partial deltas, so a gate that simply does not mention
        `assessment_report` leaves the pre-gate report in state for a checkpoint
        or a downstream reader to pick up. `assert not result.get(field)` would
        pass on exactly that defect, so both properties are asserted here.
        """
        from src.nodes.post_process_node import OUTPUT_BEARING_FIELDS

        leaky = "CREDIT UNDERWRITING ASSESSMENT\ntoken=sk-abcdefghij0123456789ABCDEF"
        result = self.node(
            {
                "assessment_report": leaky,
                "generated_answer": "ANSWER TEXT",
                "answer_citations": '[{"id": "UW-DTI-01"}]',
                "grounded": True,
                "caller_trust_level": TrustLevel.ANONYMOUS.value,
            }
        )
        cleared = {
            "assessment_report": "",
            "generated_answer": "",
            "answer_citations": "[]",
            "grounded": False,
        }
        assert set(OUTPUT_BEARING_FIELDS) == set(cleared)
        for field, empty in cleared.items():
            assert field in result, f"{field} must be present in the delta, not omitted"
            assert result[field] == empty, f"{field} must be cleared, got {result[field]!r}"

    def test_output_gate_inventory_covers_the_state_schema(self):
        """A new output-bearing State field cannot quietly stay out of the set.

        Scope is the fields THIS template adds to AgentState; the framework owns
        its own. Every one of them is either cleared on a violation, overwritten
        with the notice, or listed here as inert — adding a field without
        deciding which fails this test.
        """
        from framework.schemas.agent_state import AgentState
        from src.nodes.post_process_node import NOTICE_FIELDS, OUTPUT_BEARING_FIELDS
        from src.schemas.state import State

        # Inert: caller input echoes back to the caller's own request, retrieval
        # provenance, and the declared runtime parameters. None of them carries
        # the assessment text or a caller-facing payload.
        inert = {
            "validated_input",
            "enriched_context",
            "credit_query",
            "retrieved_documents",
            "retrieval_metadata",
            "reranked_documents",
            "top_k",
            "score_threshold",
            "rerank_keep",
            "trace_id",
            "correlation_id",
        }
        template_fields = set(State.__annotations__) - set(AgentState.__annotations__)
        uncovered = template_fields - inert - set(OUTPUT_BEARING_FIELDS) - set(NOTICE_FIELDS)
        assert not uncovered, f"State fields neither cleared nor classified inert: {sorted(uncovered)}"
        # And the inert list may not quietly absorb a field that IS cleared.
        assert not inert & set(OUTPUT_BEARING_FIELDS)

    def test_security_gate_output_helper_detects_and_clears(self):
        from src.nodes.post_process_node import _security_gate_output

        assert _security_gate_output("sk-abcdefghij0123456789ABCDEF") is not None
        assert _security_gate_output("Bearer abcdefgh12345678") is not None
        assert _security_gate_output("password = supersecret123") is not None
        assert _security_gate_output("A perfectly clean underwriting assessment.") is None

    def test_trust_level_anonymous(self):
        assert self.node.required_trust_level == TrustLevel.ANONYMOUS


# ── Graph wiring: outer AgentBaseGraph + inner BaseGraph (Cat 2 nested) ─────────


class TestOuterGraphComposition:
    def test_registers_five_backbone_slots(self):
        from src.graph.graph import (
            CreditAssessmentGraphNode,
            CreditScoringUnderwritingAssessmentReportAgent,
        )
        from src.nodes.post_process_node import PostProcessNode
        from src.nodes.pre_process_node import PreProcessNode

        agent = CreditScoringUnderwritingAssessmentReportAgent()
        agent.compile()
        assert set(agent._nodes.keys()) == {
            "initialize",
            "pre_process",
            "main",
            "post_process",
            "finalize",
        }
        assert isinstance(agent._nodes["pre_process"], PreProcessNode)
        assert isinstance(agent._nodes["main"], CreditAssessmentGraphNode)
        assert isinstance(agent._nodes["post_process"], PostProcessNode)

    def test_name_and_state_schema(self):
        from src.schemas.state import State
        from src.graph.graph import CreditScoringUnderwritingAssessmentReportAgent

        agent = CreditScoringUnderwritingAssessmentReportAgent()
        assert agent.name == "CreditScoringUnderwritingAssessmentReportAgent"
        assert agent.state_schema is State

    def test_graph_alias_matches_real_class(self):
        from src.graph.graph import Graph, CreditScoringUnderwritingAssessmentReportAgent

        assert Graph is CreditScoringUnderwritingAssessmentReportAgent

    def test_agent_class_matches_config_yaml(self):
        """The flat manifest declares ONE dotted import path, not module:/class:."""
        import re
        from pathlib import Path
        from src.graph.graph import CreditScoringUnderwritingAssessmentReportAgent

        yaml_text = (Path(__file__).resolve().parents[2] / "config" / "agent.yaml").read_text()
        assert not re.search(r"^\s*module:", yaml_text, re.M), "the flat manifest must not split module:/class:"
        m = re.search(r'^class:\s*"?([\w.]+)"?\s*$', yaml_text, re.M)
        assert m, "config/agent.yaml must declare a root-level dotted class:"
        assert m.group(1) == ("src.graph.graph." + CreditScoringUnderwritingAssessmentReportAgent.__name__)

    def test_package_reexports_agent_class_for_registry(self):
        # The class must stay importable from the src.graph PACKAGE as well as
        # from the module the manifest names.
        import src.graph as graph_pkg
        from src.graph.graph import CreditScoringUnderwritingAssessmentReportAgent

        assert hasattr(graph_pkg, "CreditScoringUnderwritingAssessmentReportAgent")
        assert graph_pkg.CreditScoringUnderwritingAssessmentReportAgent is (
            CreditScoringUnderwritingAssessmentReportAgent
        )
        assert graph_pkg.Graph is CreditScoringUnderwritingAssessmentReportAgent

    def test_parent_config_forwards_declared_settings(self):
        # _parent_config() must load config/config.yaml — NOT the flat manifest,
        # which no longer carries an `agent.config` block — and forward the
        # declared retrieval settings under `configurable`.
        from src.graph.graph import CreditAssessmentGraphNode

        configurable = CreditAssessmentGraphNode()._parent_config()["configurable"]
        assert configurable["top_k"] == 5
        assert configurable["score_threshold"] == 0.1
        assert configurable["rerank_keep"] == 3

    def test_main_slot_graphnode_contracts(self):
        from src.graph.graph import CreditAssessmentGraphNode

        node = CreditAssessmentGraphNode()
        assert node.error_strategy == "propagate"
        assert node.propagate_hitl is False
        # extract_input prefers validated_input, falls back to user_input
        assert node.extract_input({"validated_input": "V", "user_input": "U"}) == "V"
        assert node.extract_input({"user_input": "U"}) == "U"

    def test_merge_output_maps_subresult_keys(self):
        from src.graph.graph import CreditAssessmentGraphNode

        node = CreditAssessmentGraphNode()
        sub_result = {
            "assessment_report": "REPORT",
            "generated_answer": "ANSWER",
            "answer_citations": "[]",
            "grounded": True,
            "result": "REPORT",
            "status": AgentStatus.SUCCESS.value,
            "node_history": ["x"],  # not forwarded by merge_output
        }
        delta = node.merge_output({}, sub_result)
        assert delta["assessment_report"] == "REPORT"
        assert delta["grounded"] is True
        assert delta["status"] == AgentStatus.SUCCESS.value
        assert set(delta.keys()) == {
            "assessment_report",
            "generated_answer",
            "answer_citations",
            "grounded",
            "result",
            "status",
        }

    def test_agent_s3_output_gate_declared_on_class(self):
        from src.graph.graph import CreditScoringUnderwritingAssessmentReportAgent

        agent = CreditScoringUnderwritingAssessmentReportAgent()
        assert agent._security_gate_output("sk-abcdefghij0123456789ABCDEF") is not None
        assert agent._security_gate_output("A clean advisory assessment.") is None


class TestInnerDomainGraph:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        for mod in (
            "input_validate_node",
            "retrieve_node",
            "rerank_filter_node",
            "generate_answer_node",
            "output_format_node",
        ):
            monkeypatch.setattr(f"src.nodes.{mod}.emit_trace_event", lambda *a, **k: None)

    def test_registers_five_domain_nodes(self):
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        g = DomainWorkflowGraph()
        g.register_nodes()
        assert set(g._nodes.keys()) == {
            "input_validate",
            "retrieve",
            "rerank_filter",
            "generate_answer",
            "output_format",
        }

    def test_name_and_state_schema(self):
        from src.schemas.state import State
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        g = DomainWorkflowGraph()
        assert g.name == "fin_c2_066_credit_underwriting_assessment_workflow"
        assert g.state_schema is State

    def test_extra_initial_state_seeds_forwarded_config(self):
        # Forwarded config on self.config["configurable"] is copied into the
        # inner initial state so the RAG nodes observe it on the .invoke() path.
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        g = DomainWorkflowGraph(config={"configurable": {"top_k": 2, "score_threshold": 0.3, "rerank_keep": 1}})
        seeded = g._extra_initial_state()
        assert seeded == {"top_k": 2, "score_threshold": 0.3, "rerank_keep": 1}

    def test_extra_initial_state_empty_without_config(self):
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        assert DomainWorkflowGraph()._extra_initial_state() == {}

    def test_inner_graph_invoke_produces_grounded_assessment(self):
        """Standalone inner-graph invoke (ANONYMOUS caller) runs the linear RAG pipeline
        and shapes the get_output() dict consumed by the outer merge_output()."""
        from framework.schemas.invocation_context import InvocationContext, TrustLevel
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        g = DomainWorkflowGraph()
        g.compile()
        ctx = InvocationContext(caller_trust_level=TrustLevel.ANONYMOUS)
        result = g.invoke(GROUNDED_QUERY, ctx=ctx)
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["assessment_report"] is not None
        assert result["grounded"] is True
        assert "CREDIT UNDERWRITING ASSESSMENT" in result["assessment_report"]

    def test_inner_graph_abstains_on_offtopic_query(self):
        """An off-topic query retrieves nothing -> grounded abstention (still SUCCESS)."""
        from framework.schemas.invocation_context import InvocationContext, TrustLevel
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        g = DomainWorkflowGraph()
        g.compile()
        ctx = InvocationContext(caller_trust_level=TrustLevel.ANONYMOUS)
        result = g.invoke(OFFTOPIC_QUERY, ctx=ctx)
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["grounded"] is False


class TestOuterGraphGetOutput:
    """SR re-review (2026-07): the compiled OUTER graph must surface the structured
    domain result on a successful invoke — not just the base {output} envelope.

    Without a get_output() override, AgentBaseGraph.get_output() returns only
    {output, status, trace_id, correlation_id, node_history}; the domain keys
    that CreditAssessmentGraphNode.merge_output() adds (generated_answer,
    answer_citations, grounded, assessment_report, result) would be dropped
    (None) to the caller. This regression invokes the compiled outer graph and
    asserts they are present on the S-3-gated success path.
    """

    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        for mod in (
            "pre_process_node",
            "input_validate_node",
            "retrieve_node",
            "rerank_filter_node",
            "generate_answer_node",
            "output_format_node",
            "post_process_node",
        ):
            monkeypatch.setattr(f"src.nodes.{mod}.emit_trace_event", lambda *a, **k: None)

    def test_compiled_outer_invoke_surfaces_domain_result(self):
        from framework.schemas.invocation_context import InvocationContext, TrustLevel
        from src.graph.graph import Graph

        agent = Graph()
        agent.compile()
        ctx = InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL)
        result = agent.invoke(GROUNDED_PAYLOAD, ctx=ctx)

        assert result["status"] == AgentStatus.SUCCESS.value
        # Base envelope preserved.
        assert result.get("output") is not None
        # Domain result surfaced by the get_output() override (was dropped before).
        assert result.get("grounded") is True
        assert result.get("generated_answer") is not None
        assert result.get("result") is not None
        assert "CREDIT UNDERWRITING ASSESSMENT" in result["result"]
        cites = from_json(result.get("answer_citations"))
        assert any(c.get("id") == "UW-DTI-01" for c in cites)
