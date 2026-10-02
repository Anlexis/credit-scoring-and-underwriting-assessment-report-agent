# FIN-C2-066 — Credit Underwriting Assessment Agent

> **Category**: Cat 2 (retrieval-augmented generation)
> **Industry**: FIN

## Overview

Answers a credit-underwriting question by retrieving the underwriting-policy passages that
apply to it and assembling a cited, advisory assessment from those passages alone. The agent
returns the policy criteria a human underwriter needs — debt-to-income caps, credit-score
bands, collateral and loan-to-value rules, income-verification requirements, adverse-history
triggers — each attributed to the passage it came from, followed by a mandatory advisory
disclaimer and a `Grounded in retrieved policy: YES/NO` line. When no passage clears the
relevance threshold it abstains and says so instead of answering from general knowledge.

The pipeline is deterministic in this version: retrieval, reranking and answer synthesis are
implemented as explicit, testable steps over a small in-repository policy corpus, and there is
**no model call**. `prompts/credit_underwriting_assessment.j2` and the `llm` block in
`config/config.yaml` document the generation contract for wiring a real model and retriever in;
the grounding rule — every statement traceable to a cited passage — is the same either way.

It does **not** score an applicant, rank against portfolio thresholds, or render any computed
monetary figure. It is decision support for the person who does, and the output says so.

This is an agent template built with the **AGENTIC STAR** development platform and the
**AgentCore Framework**. It is intended to be taken as a starting point: fork it, adapt it to
your own data and policies, and run it inside your own AGENTIC STAR deployment.

## Requirements

**This template does not run standalone.** It requires:

| Requirement | Notes |
|---|---|
| **AGENTIC STAR platform** | The agent connects to the platform at start-up. Without it, start-up fails immediately (see *Behaviour without the platform* below). Deployment guides and API documentation: [AGENTIC STAR Developers](https://developers.fd.agenticstar.tm.softbank.jp/) |
| **AgentCore Framework** (`agenticstar-agentcore`) | Installed from PyPI as a dependency. |
| Python | >=3.11 |

```bash
pip install -e .
```

### Behaviour without the platform

The framework is designed to run **only** on AGENTIC STAR. There is no fallback or degraded
mode: if the platform is unreachable or the framework version does not match, graph compile
and start-up preflight raise rather than starting in a partially working state. This is
intentional — a half-running agent is worse than one that refuses to start.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests run without a platform connection. Running the agent itself does not.

## Calling it

`POST /invoke` takes the question as `input`, either as plain text or as a JSON object with a
`query` field:

```json
{ "input": "{\"query\": \"What debt-to-income ratio applies to an unsecured consumer loan?\"}" }
```

Set `INVOKE_AUTH_TOKEN` in the server environment to require `Authorization: Bearer <token>`
on callers the platform has not already vouched for; without it those callers are anonymous and
the input trust gate refuses them. A successful response carries the assembled assessment in
`output`, the cited passage ids in `answer_citations`, and `grounded`. On any refusal the
response carries a short notice and nothing else — no partial answer, no diagnostic text.

## Configuration

`config/agent.yaml` is the static manifest (identity, entry-point class, required trust level,
and the secrets/extras the agent needs at compile time — both empty here). `config/config.yaml`
holds the runtime parameters, and they are live: `retrieval.top_k`, `retrieval.score_threshold`
and `retrieval.rerank_keep` change how many passages are retrieved, which of them clear the
relevance bar, and how many are cited. Values outside their documented range, or non-finite
ones, are rejected in favour of the node defaults rather than silently disabling the filter.

## Project Structure

```
src/          agent implementation (nodes, graphs, security screens, schemas)
tests/        unit and boundary tests
config/       agent manifest and runtime parameters
prompts/      generation prompt template
docs/         design and operational documentation
```

See `docs/` for the design and the test specification.

## Customising

1. Adjust `config/config.yaml` for your own retrieval and generation parameters.
2. Replace `RetrieveNode`'s in-module policy corpus with your own retriever and policy set.
3. Review the node implementations under `src/nodes/` for domain-specific logic — in
   particular `src/security/screens.py`, which holds the input and output screens.
4. Re-run the test suite.

## License

MIT — see [LICENSE](LICENSE).

## Status of this repository

This template is published **as is**, by its individual author, under the MIT license. It carries
**no warranty and no support commitment**, and no organisation stands behind its behaviour or
fitness for any purpose. Issues and pull requests may or may not receive a response; that is at
the sole discretion of the repository owner.
