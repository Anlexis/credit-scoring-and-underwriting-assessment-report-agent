# Template Design Specification — FIN-C2-066

Credit Scoring & Underwriting Assessment Report Agent — **Cat 2 (RAG)**, nested two-layer architecture.

## Position in AgentCore Architecture

- **Agent Class**: `CreditScoringUnderwritingAssessmentReportAgent`
- **L1 Base**: `AgentBaseGraph` (direct L1 inheritance; **not** `AutonomousBaseGraph`)
- **Pattern**: Cat 2 RAG — retrieval-grounded underwriting-policy assessment (RAGAgent base type)
- **Three-Layer Separation**:
  - State: flat TypedDict composition (no Pydantic — msgpack incompatible)
  - Node: L1 inheritance (Template Method: `execute(self, state) -> dict` override only)
  - Graph: composition (`register_nodes()` for node substitution; nested `GraphNode` subgraph)

## Architecture Overview

This template uses the Cat 2 **two-layer nested** pattern. The outer graph is the
fixed 5-node backbone (`AgentBaseGraph`); its `main` slot is a `GraphNode`
(`CreditAssessmentGraphNode`) that delegates the full RAG pipeline to an inner
`BaseGraph` (`DomainWorkflowGraph`).

### Node Configuration

| Node | Layer | Responsibility | Input State | Output State | Inherits/Overrides |
|------|-------|----------------|-------------|--------------|--------------------|
| initialize | outer | schema/session/trust init | user_input | session_id, trust_level | InitializeNode (default) |
| pre_process | outer | **S-1** trust gate (VERIFIED_EXTERNAL) + **S-2** domain input screens + query validation | user_input | validated_input, enriched_context | PreProcessNode |
| main | outer | delegate to inner RAG graph | validated_input | assessment_report, generated_answer, answer_citations, grounded | CreditAssessmentGraphNode (GraphNode) |
| post_process | outer | **S-3** output gate + clear-on-violation + expose result | assessment_report | formatted_output, result, cleared output-bearing fields | PostProcessNode |
| finalize | outer | response metadata | result | response_metadata | FinalizeNode (default) |
| input_validate | inner | domain query validation/normalisation | validated_input | credit_query | InputValidateNode |
| retrieve | inner | top_k policy-passage retrieval | credit_query | retrieved_documents, retrieval_metadata | RetrieveNode |
| rerank_filter | inner | rerank + score_threshold filter | retrieved_documents | reranked_documents | RerankFilterNode |
| generate_answer | inner | grounded answer synthesis (LLM in prod) | reranked_documents | generated_answer, answer_citations, grounded | GenerateAnswerNode |
| output_format | inner | assemble advisory assessment | generated_answer, answer_citations | assessment_report, result | OutputFormatNode |

### Data Flow

```
Outer backbone:
  START → initialize → pre_process → main → {route} → post_process → finalize → END
                                        ↓ (retry)
                                      pre_process

main (CreditAssessmentGraphNode) → inner DomainWorkflowGraph:
  START → input_validate → retrieve → rerank_filter → generate_answer → output_format → END
```

### State Definition

| Field | Type | Purpose | Required |
|-------|------|---------|----------|
| validated_input | NotRequired[Optional[str]] | normalised query (pre_process) | no |
| enriched_context | NotRequired[Optional[str]] | JSON channel metadata | no |
| credit_query | NotRequired[Optional[str]] | canonical query | no |
| retrieved_documents | NotRequired[Optional[str]] | JSON list of {id,title,text,score} | no |
| retrieval_metadata | NotRequired[Optional[str]] | JSON retrieval stats | no |
| reranked_documents | NotRequired[Optional[str]] | JSON filtered passages | no |
| generated_answer | NotRequired[Optional[str]] | grounded answer text | no |
| answer_citations | NotRequired[Optional[str]] | JSON list of {id,title} | no |
| grounded | NotRequired[Optional[bool]] | grounded-in-KB flag | no |
| assessment_report | NotRequired[Optional[str]] | final advisory assessment | no |
| result | NotRequired[Optional[str]] | primary caller result | no |

**State Constraints (mandatory):**
- Flat TypedDict only (primitives + JSON-serialized `Optional[str]`; `to_json`/`from_json` at every producer/consumer)
- No JWT, API keys, credentials in State (checkpoint DB leakage)
- Applicant credit data = individual credit information (個人情報, APPI) — never persisted beyond the invocation
- InvocationContext via `config["configurable"]` only (not in State)
- No Pydantic models, dataclass, arbitrary Python objects (msgpack incompatible)
- `formatted_output` is inherited from `AgentState` — NOT re-declared

## Output Invariant

**There is no monetary precision grid here, and that is a decision, not an omission.**
This template renders no computed monetary aggregate: the report is assembled from
retrieved policy passages and the caller's own question, and no node performs
arithmetic on an amount. A rounding gate over that text would have nothing correct to
round and would instead rewrite the figures, dates and statutory citations the passages
carry — a numeric snap applied to `第3条`, `2026-09-04` or `100,000,000` corrupts them.

The invariant enforced instead is the opposite one, and it is the one the generation
prompt already states: **figures and citations reach the report unaltered, and personal
identifiers never reach it at all.**

| Invariant | Where enforced | Where proved |
|---|---|---|
| Numbers, dates, percentages, ratios and statutory citations are byte-identical between input and rendered report | no numeric rewriting anywhere in the render path | `TestNumericFidelity` (E2E) + `test_numbers_dates_and_citations_are_byte_identical` |
| Policy-passage ids (`UW-DTI-01` …) render verbatim | `OutputFormatNode._assemble` | `test_policy_passage_ids_are_rendered_verbatim` |
| Personal identifiers in the caller's question never reach the report | `PreProcessNode._extra_security_gate_input` → `mask_payload_pii` | `TestInputScreensEndToEnd` |
| Caller text cannot manufacture a cited policy line | whitespace normalisation in `InputValidateNode` | `test_caller_text_cannot_manufacture_a_cited_step` |

## Security Layers Beyond the Framework Defaults

The framework's own gates leave three gaps that matter for a Japanese financial
template. Each is closed once, in one place, so that removing it is observable
end to end rather than masked by a second layer.

| Gap | Measured behaviour | Closed by |
|---|---|---|
| The injection detector blocks the literal `<\|im_start\|>` and the bracketed `[INST]`/`[SYS]` forms, but not the class | `<<SYS>>`, `<\|system\|>` and `<\|start_header_id\|>` were echoed into a report claiming grounding | `screen_control_tokens` — scans raw, markup-stripped, and every parsed string leaf and mapping key; fail closed |
| The PII detector delimits its digit-group patterns with a word boundary, and Kana/Kanji are word characters | `個人番号1234-5678-9012を確認` rendered verbatim in the caller-facing report while `My number is 1234-5678-9012` masked. Japanese is written without spaces, so the failing case is the normal one | `mask_residual_pii` — re-runs the **framework's own** detector over a copy with a separator inserted at each script boundary, so the block sets stay identical by construction |
| The credential detector describes credential formats only | the local gate's `password: …` pattern catches what the framework misses; the framework's `AKIA`, `sk_live_` and connection-string patterns catch what the local set missed | `detect_output_credentials` — the **union**, never a replacement in either direction |

### Envelope containment

`AgentBaseGraph.get_output` resolves `formatted_output or result` with no status check,
and `BaseNode.__call__` converts an exception inside post_process into a bare ERROR
partial that clears nothing. Both halves are therefore fixed:

* `PostProcessNode` returns a **truthy** notice and explicitly clears every
  output-bearing field. A falsy replacement re-opens the fallback, and an omitted key
  leaves the pre-gate value in state for a checkpoint to pick up.
* `CreditScoringUnderwritingAssessmentReportAgent.get_output` surfaces `result` **only**
  on the gated success path; on any non-success it re-resolves the envelope from
  `formatted_output` alone.

## Framework Utilization

### Shared Components Used
- [x] S-1 trust gate — `PreProcessNode.required_trust_level = VERIFIED_EXTERNAL` (inner domain nodes ANONYMOUS)
- [x] S-2 input screens — `PreProcessNode._extra_security_gate_input()` (the framework's domain hook)
- [x] S-3 output gate — `_security_gate_output` declared on the agent class (the framework rules §5-6); runtime enforcement in `PostProcessNode` via the same module-level screen
- [x] S-4 audit — `emit_trace_event()` (positional) inside every node `execute()`
- [x] `GraphNode` composition — `CreditAssessmentGraphNode` wraps the inner `DomainWorkflowGraph`

> **S-2/S-3 gate behaviour by node type (ADR-017):** the domain S-3 scan runs as a
> module-level function inside `PostProcessNode.execute()` (not as an `_extra_security_gate_output`
> instance method) to avoid the real-SDK auto-wrap None-state AttributeError (CoE a peer template / a peer template).
> The agent-class declaration and the runtime enforcement call the SAME function
> (`src/security/screens.detect_output_credentials`) so they cannot drift apart.

### Runtime configuration contract

`config/agent.yaml` is the FLAT manifest — AgentRegistry reads every key at root level
and there is no `agent:` block, so the runtime parameters live in `config/config.yaml`
and `_parent_config()` reads them from there. `requires.secrets` and `requires.extras`
are both `[]`: the template calls no `ctx.secrets.require()` and constructs no client,
and declaring an unprovisioned secret would make the agent fail at compile time.

Declared values are parsed **finite and bounded** before they are seeded
(`top_k` 1–50, `score_threshold` 0.0–1.0, `rerank_keep` 1–50). `float("NaN")` parses
cleanly and compares False against every threshold, so an unvalidated value would drop
every retrieved passage and return a confident "no policy applies" answer.

### Composition Pattern

- **Pattern**: nested `GraphNode` subgraph (Cat 2 two-layer)
- **Composition target**: `DomainWorkflowGraph` (inner `BaseGraph`)
- **Error propagation strategy**: `propagate` (fail-fast; inner errors re-raised as SubgraphError)

## Import Isolation Confirmation
- [x] Template does not import agenticstar-platform SDK (Level 0)
- [x] Import targets: `framework/` and `shared/` only (no `agents/base/` required)

## Design Decision Record

| Decision | Option A | Option B | Chosen | Rationale |
|----------|----------|----------|--------|-----------|
| L1 base type | AgentBaseGraph | AutonomousBaseGraph | **AgentBaseGraph** | Fixed 5-node backbone; deterministic RAG pipeline, no autonomous loop |
| Composition pattern | Standalone nodes | Nested GraphNode subgraph | **Nested GraphNode subgraph** | Encapsulates multi-step RAG topology behind the `main` slot |
| Retrieval | Real vector store | Deterministic in-module corpus | **Deterministic** | No model or vector client is wired in this version; the retriever is wired via config in a deployment. The grounding contract is identical either way |
| Monetary precision grid | Round rendered aggregates | No grid — enforce byte fidelity | **No grid** | Nothing here computes a monetary aggregate. A numeric snap over policy text rewrites `第3条`, `2026-09-04` and `100,000,000` instead of rounding an amount. See *Output Invariant* |
| Residual PII screen | Restate the framework's patterns locally | Re-run the framework's detector across the script boundary | **Re-run the framework's detector** | A local restatement drifts from the framework's block set; re-running it keeps the two identical by construction |
