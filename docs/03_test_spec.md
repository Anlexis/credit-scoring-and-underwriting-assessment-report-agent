# Test Specification — FIN-C2-066 Credit Scoring & Underwriting Assessment Report Agent

## 1. Test Strategy

- **Agent:** FIN-C2-066 — Credit Scoring & Underwriting Assessment Report Agent
  (Cat 2, RAG pattern, two-layer nested graph: outer `AgentBaseGraph` backbone +
  inner `DomainWorkflowGraph` `BaseGraph`).
- **Coverage target:** ≥ 90% of `src/nodes/` + `src/graph/` branches.
- **Test types:** Unit (per node + graph wiring + input/output screens) ·
  Proof-of-Boundary (framework security/serialization contracts) · Backbone invoke
  (full `Graph().invoke()`) · Integration end-to-end through the real ASGI
  `POST /invoke` with a bearer credential.
- **Framework provisioning:** `framework` (agenticstar-agentcore) is installed by
  CI from the package registry — wheel only. Tests import the real modules; there
  are no stub nodes.
- **S-4 audit:** `emit_trace_event` is patched at the node module level in unit
  tests to avoid audit-backend calls, never via a `sys.modules` stub (which would
  break the real `shared` package the framework loads at import time).

### Test file map

| File | Scope |
|------|-------|
| `tests/unit/test_nodes.py` | All 7 domain/backbone nodes (pre_process · input_validate · retrieve · rerank_filter · generate_answer · output_format · post_process) + outer & inner graph wiring + the output-gate clearing inventory |
| `tests/unit/test_input_screens.py` | `src/security/screens.py` — control-token class, residual-PII masking, credential-screen union, finite/bounded parsing |
| `tests/unit/test_framework_compliance_tc06_tc07.py` | TC-06/TC-07 — the framework's S-2/S-3 gates are not overridable |
| `tests/integration/test_invoke_e2e.py` | End-to-end through the real ASGI `POST /invoke`: auth, servable path, declared-config liveness, input screens, numeric fidelity, envelope containment |
| `tests/proof_of_boundary/test_pb_invoke_order.py` | PB-6 per-node + backbone invoke order (VERIFIED_EXTERNAL) + S-1 node-level gate + payload alignment |
| `tests/proof_of_boundary/test_import_isolation.py` | PB-4 Level-0 import isolation (AST scan) |
| `tests/proof_of_boundary/test_state_safety.py` | PB-2/PB-5 State msgpack/credential safety (AST scan) |
| `tests/proof_of_boundary/test_pb7_hitl_interrupt_propagation.py` | PB-7 HITL interrupt-propagation (skip stub — no cross-boundary HITL) |

### Canonical valid payload (PB-6 `_VALID_PAYLOAD`)

The grounded underwriting query used by the backbone invoke test and by
`deploy/invoke_payload.json` (the two MUST stay identical — asserted by
`test_invoke_payload_matches_pb6`):

```json
{
  "query": "What debt-to-income (DTI) ratio and credit-score band thresholds apply when underwriting an unsecured consumer credit application, and what income verification is required?"
}
```

Grounding: the query overlaps `RetrieveNode._KB_CORPUS` passages **UW-DTI-01**
(DTI threshold), **UW-SCORE-02** (credit-score bands) and **UW-INCOME-04**
(income verification) above the `score_threshold` (0.1), so retrieval is
non-empty, `RerankFilterNode` keeps the grounding passages, `GenerateAnswerNode`
sets `grounded = True`, and the outer S-3 gate passes ⇒ `status = success`.

## 2. Framework Compliance Tests (Mandatory)

| TC-ID | Test | Expected Result | Where |
|-------|------|----------------|-------|
| TC-01 | State contract: flat `TypedDict`, domain fields `NotRequired`, no Pydantic/dataclass | AST scan: 0 violations | `test_state_safety.py` |
| TC-02 | Empty / query-less input rejected at PreProcessNode | `status=error`, error_log populated | `TestPreProcessNode` |
| TC-03 | No JWT/credential in State | CI `gate-credential-scan`: 0 violations | CI + `test_state_safety.py` |
| TC-04 | `execute(self, state)` contract — no `_invoke_impl` / `process` | Signature `(self, state)` on EVERY discovered node | `test_execute_signature_is_state_first` |
| TC-05 | S-4: `emit_trace_event()` called inside each node `execute()` | ≥1 domain event per node (positional form) | verified by CoE preflight S-4/#3 |
| TC-08 | S-1: `required_trust_level` enforced in `__call__` before `execute()` | ANONYMOUS caller → refused at node; VERIFIED_EXTERNAL → admitted | `TestS1TrustGate` |
| TC-08a | Outer `PreProcessNode` = VERIFIED_EXTERNAL; inner nodes + post_process = ANONYMOUS | trust levels asserted per node | `test_trust_level_*` |
| TC-09 | S-2 domain input screen: chat-template control tokens | `<<SYS>>` / `<|…|>` / `[INST]` refused raw, markup-stripped and post-parse (keys included) | `TestControlTokenScreen`, `test_a_control_token_is_refused_and_never_echoed` |
| TC-10 | S-2 residual PII in unspaced Japanese | `個人番号1234-5678-9012を確認` masked; figures/dates/citations byte-identical | `TestResidualPiiScreen`, `TestInputScreensEndToEnd` |
| TC-11 | S-3 output gate on post_process | credential class → truthy notice + `status=error` + every output-bearing field cleared; clean → pass | `TestPostProcessNode` |
| TC-11a | S-3 screen is never narrower than the framework detector | every framework pattern also trips the local screen; local `password:` pattern still fires | `TestOutputCredentialScreen` |
| TC-12 | Agent-class S-3 gate declared (`_security_gate_output`) | detects credential patterns; clears clean text | `test_agent_s3_output_gate_declared_on_class` |
| TC-13 | Flat manifest declares ONE dotted `class:` and no `module:` | exact string match | `test_agent_class_matches_config_yaml` |
| TC-14 | Declared runtime config is LIVE end-to-end | changing `rerank_keep` / `score_threshold` changes the citations returned by `/invoke` | `TestDeclaredConfigIsLive` |
| TC-15 | Non-finite declared config fails closed | `score_threshold: NaN` falls back to the default instead of dropping every passage | `test_a_non_finite_threshold_falls_back_instead_of_disabling_the_filter` |
| TC-16 | Envelope containment on a non-success path | a data-path fault in `merge_output` must not surface the pre-gate report, a traceback or a source path | `TestEnvelopeContainment` |

## 3. Proof-of-Boundary Tests (Mandatory)

| PB-ID | Boundary | Test | Expected Result | Where |
|-------|----------|------|----------------|-------|
| PB-2 | State serialization | AST scan of `src/schemas/state.py` | primitives only; no Pydantic/dataclass | `test_state_safety.py` |
| PB-4 | Import isolation | AST scan of `src/` | 0 Level-0 (`agenticstar` / platform) imports | `test_import_isolation.py` |
| PB-5 | Checkpoint safety | no credential-named fields / prohibited types in State | inspection pass | `test_state_safety.py` |
| PB-6 | Invoke execution order (per node) | `__call__`: S-4 node_start → S-1 gate → S-2 input gate → `execute()` → S-3 output gate → S-4 node_complete | order verified for every `src/nodes/` class | `TestInvokeOrder` |
| PB-6b | Backbone invoke order | full `Graph().invoke(_VALID_PAYLOAD, ctx=VERIFIED_EXTERNAL)` | `status=success`; node_history = `[Initialize, PreProcess, CreditAssessmentGraphNode, PostProcess, Finalize]` | `TestBackboneInvokeOrder` |
| PB-6c | Real external caller | `InvocationContext(caller_trust_level=VERIFIED_EXTERNAL)` — **never** `for_internal()` | inner ANONYMOUS nodes accept the passthrough trust; SUCCESS end-to-end | `TestBackboneInvokeOrder` |
| PB-6d | Payload alignment | `deploy/invoke_payload.json["input"] == _VALID_PAYLOAD` | Stage-5 deploy-stg invoke exercises the PB-6 payload | `test_invoke_payload_matches_pb6` |
| PB-7 | HITL interrupt propagation | skip stub — `propagate_hitl=False`, no cross-boundary interrupt() checkpoint | skipped with reason (real assertion when HITL wired) | `test_pb7_hitl_interrupt_propagation.py` |
| PB-8 | Real HTTP entry point | `POST /invoke` at the declared trust level, bearer credential | 401 without a credential; SUCCESS with one; identical refusal body for absent vs wrong | `TestAuthentication`, `TestServablePath` |
| PB-9 | No caller context channel | the request model declares `input` + `session_id` only; an unknown context field is dropped, not forwarded to `invoke()` | a credential-shaped value in an unknown field cannot reach the first node | `TestNoContextChannel` |

## 4. Business Logic Tests (RAG grounding + abstention)

| BL-ID | Test | Input | Expected Result | Where |
|-------|------|-------|----------------|-------|
| BL-01 | Happy-path grounded assessment | `_VALID_PAYLOAD` | advisory assessment; `CREDIT UNDERWRITING ASSESSMENT` + `Grounded in retrieved policy: YES` present | `test_backbone_invoke_succeeds_and_returns_output`, `test_inner_graph_invoke_produces_grounded_assessment` |
| BL-02 | Query extraction (plain string / JSON object) | `"What DTI…"` / `{"query": "…"}` | both yield `validated_input` = the query | `test_plain_string_query_returns_success`, `test_json_object_query_is_extracted` |
| BL-03 | Retrieval grounding | DTI query | retrieves `UW-DTI-01` (+ others) above threshold | `test_grounded_query_retrieves_passages` |
| BL-04 | **Abstention** — off-topic query | `"zzz qqq wibble…"` | retrieval empty; `grounded=False`; no-answer text; `status=success` | `test_offtopic_query_retrieves_nothing`, `test_abstains_when_no_grounding`, `test_inner_graph_abstains_on_offtopic_query` |
| BL-05 | Rerank + threshold filter | scored candidates (0.5 / 0.2 / 0.05) | 0.05 dropped (< 0.1); top-N kept in score order | `test_reranks_and_filters_by_threshold`, `test_keep_limit_is_respected` |
| BL-06 | Grounded answer + citations + disclaimer | reranked passages | `grounded=True`; cites `UW-DTI-01`; advisory disclaimer present | `test_grounded_answer_cites_passages`, `test_answer_carries_advisory_disclaimer` |
| BL-07 | Report assembly | answer + citations + grounded flag | header + `CITED UNDERWRITING-POLICY PASSAGES` + `Grounded in retrieved policy: YES/NO` | `TestOutputFormatNode` |
| BL-08 | Graph key coupling | inner `get_output` ↔ outer `merge_output` | 6 coupled keys mapped; `merge_output` returns changed keys only | `test_merge_output_maps_subresult_keys`, `TestInnerDomainGraph` |
| BL-09 | Cat 2 composition | outer 5 backbone slots + inner 5 domain nodes | correct slot/node registration; `main` slot is a `GraphNode` | `TestOuterGraphComposition`, `test_registers_five_domain_nodes` |

### Negative / boundary cases

| Case | Node | Expected |
|------|------|----------|
| empty `user_input` | PreProcessNode | `status=error`, "empty" |
| JSON object with no query key | PreProcessNode | `status=error`, "query" |
| overlong query (> 4000 chars) | PreProcessNode | `status=error`, "too long" — never truncated |
| structured payload that does not parse | PreProcessNode | `status=error`; the payload is NOT echoed back |
| chat-template control token | PreProcessNode | `status=error`; nothing echoed |
| empty query | InputValidateNode | `status=error`, "empty" |
| query < 3 chars | InputValidateNode | `status=error`, "short" |
| missing `credit_query` | RetrieveNode | `status=error`, "credit_query" |
| no KB overlap | RetrieveNode | empty retrieval, `status=success` (abstention) |
| empty retrieval | RerankFilterNode | empty reranked, `status=success` |
| no grounding passages | GenerateAnswerNode | no-answer text, `grounded=False`, `status=success` |
| missing `generated_answer` | OutputFormatNode | `status=error`, "generated_answer" |
| empty `assessment_report` | PostProcessNode | fallback message, `status=success` |
| credential leak in output | PostProcessNode | truthy withholding notice, `status=error`, every output-bearing field cleared (S-3) |
| post_process produces no output | agent `get_output()` | envelope re-resolved from `formatted_output` alone — `result` is never surfaced on a non-success path |

## 5. Test Execution Summary

- Execution: `pytest tests/` against the installed framework wheel.
- Total: 173 tests — **172 passed, 1 skipped** (PB-7 skip stub, by design).
- Gates: `gate-dep-pinning`, `gate-stub-check`, `gate-cat-consistency`,
  `gate-audit-trace-check`, `gate-manifest-schema`, `gate-oss-license`,
  `gate-forbidden-strings`, import-isolation, composition, invoke-chain,
  credential-scan, trust-level, scaffold-integrity — all PASS.
- Coverage: node + graph modules exercised on the grounded, abstention, refusal
  and containment paths, through direct calls AND the real ASGI entry point.
