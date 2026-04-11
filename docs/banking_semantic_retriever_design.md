# Banking Knowledge Semantic Retriever Design

## Status

Draft

## Problem

The current `banking_knowledge` semantic retrieval path is not good enough for long, high-density policy documents.

At the moment, the embedding-based pipeline does:

1. one embedding per full document
2. one query embedding per KB query
3. cosine similarity over full-document vectors
4. optional postprocessing

This is too coarse for the banking domain because:

- many documents are long and contain multiple unrelated subsections
- exact product names matter a lot
- many tasks are really "find the one policy snippet that changes the action"
- semantically broad documents can outrank the correct card-specific document

## Evidence

The first smoke test with the LM Studio-backed embedding model succeeded at the infrastructure level but failed at the task level:

- retrieval config: `openai_embeddings`
- embedding model: `EMBEDDING_MODEL=text-embedding-qwen3-embedding-8b`
- smoke test task: `task_001`
- reward: `0.0`

Observed failure pattern from the trajectory:

- early `KB_search` calls returned irrelevant or weakly related documents
- examples included debit-card and dispute-policy documents for a credit-card recommendation task
- the agent drifted into the wrong workflow before recovering partially
- later answers hallucinated card fee and rewards facts
- the user ultimately applied for `Diamond Elite Card`
- the expected action in `task_001` is `apply_for_credit_card(card_type="Gold Rewards Card", ...)`

This suggests the current issue is retrieval quality, not endpoint connectivity or env wiring.

## Goals

1. Improve document selection for long banking policy documents.
2. Keep the external `KB_search` tool contract unchanged.
3. Preserve compatibility with the existing Tau2 knowledge pipeline.
4. Keep retrieval deterministic and fast enough for benchmark runs.
5. Avoid overfitting to a single task while still fixing obvious card-name failures.

## Non-Goals

1. Changing the agent prompt structure for the whole domain.
2. Replacing the tool interface used by `banking_knowledge`.
3. Building a full reranking service or cross-encoder stack in the first pass.
4. Solving all benchmark failures with retrieval alone.

## Current Architecture

Current embedding-based retrieval in `banking_knowledge` is assembled in:

- `benchmarks/tau2_bench/repo/src/tau2/domains/banking_knowledge/retrieval.py`

The generic semantic stack is:

- `embedding_indexer`
- `embedding_encoder`
- `cosine` retriever

The LM Studio-backed setup now routes through the OpenAI-compatible embedding path with:

- `LMSTUDIO_BASE_URL`
- `LMSTUDIO_TOKEN`
- `EMBEDDING_MODEL`

### LM Studio Qwen Formatting Follow-Up

The local LM Studio setup currently serves the Qwen embedding model through the
OpenAI-compatible endpoint, not through the OpenRouter-specific client.

That creates a semantic mismatch:

- `OpenRouterEmbedder` already knows to apply Qwen query formatting
- `OpenAIEmbedder` previously treated all models as plain text embeddings
- the local LM Studio path therefore lost the Qwen query instruction prefix

Implementation plan:

1. Keep LM Studio on the OpenAI-compatible transport path.
2. Move Qwen query-formatting behavior into shared embedding-query semantics.
3. Teach `OpenAIEmbedder` to apply the same query formatting when the model is Qwen-family.
4. Explicitly disable query formatting in document indexers so document embeddings remain raw.
5. Extend tests to cover the OpenAI-compatible local path and cache-key behavior.

Why this approach:

- patching `OpenRouterEmbedder` to talk to LM Studio would blur provider boundaries
- creating a separate `LMStudioEmbedder` would duplicate an OpenAI-compatible transport we already have
- the actual missing behavior is model-family query formatting, not transport

Status:

- in progress

## Implemented Changes

### 1. Shared query-formatting helper

Added a shared helper module:

- [query_formatting.py](/home/hangjianli/Projects/agentBench/benchmarks/tau2_bench/repo/src/tau2/knowledge/embedders/query_formatting.py)

Key additions:

- `DEFAULT_QWEN_QUERY_INSTRUCTION`
- `model_requires_query_instruction(...)`
- `resolve_query_instruction(...)`
- `format_query_text(...)`

Relevant lines:

- [query_formatting.py](/home/hangjianli/Projects/agentBench/benchmarks/tau2_bench/repo/src/tau2/knowledge/embedders/query_formatting.py#L7)
- [query_formatting.py](/home/hangjianli/Projects/agentBench/benchmarks/tau2_bench/repo/src/tau2/knowledge/embedders/query_formatting.py#L13)
- [query_formatting.py](/home/hangjianli/Projects/agentBench/benchmarks/tau2_bench/repo/src/tau2/knowledge/embedders/query_formatting.py#L20)
- [query_formatting.py](/home/hangjianli/Projects/agentBench/benchmarks/tau2_bench/repo/src/tau2/knowledge/embedders/query_formatting.py#L36)

Why:

- move Qwen query semantics out of the OpenRouter-specific transport class
- make query formatting model-family driven instead of provider driven

### 2. OpenAI-compatible embedder now supports Qwen query formatting

Updated:

- [openai_embedder.py](/home/hangjianli/Projects/agentBench/benchmarks/tau2_bench/repo/src/tau2/knowledge/embedders/openai_embedder.py)

Code changes:

- imports shared query-formatting helpers
- accepts `prefix` and `query_instruction`
- resolves an effective query instruction from the model name
- formats query text before calling `embeddings.create(...)`
- updates `get_name()` to reflect query-formatted mode

Relevant lines:

- [openai_embedder.py](/home/hangjianli/Projects/agentBench/benchmarks/tau2_bench/repo/src/tau2/knowledge/embedders/openai_embedder.py#L10)
- [openai_embedder.py](/home/hangjianli/Projects/agentBench/benchmarks/tau2_bench/repo/src/tau2/knowledge/embedders/openai_embedder.py#L19)
- [openai_embedder.py](/home/hangjianli/Projects/agentBench/benchmarks/tau2_bench/repo/src/tau2/knowledge/embedders/openai_embedder.py#L38)
- [openai_embedder.py](/home/hangjianli/Projects/agentBench/benchmarks/tau2_bench/repo/src/tau2/knowledge/embedders/openai_embedder.py#L50)
- [openai_embedder.py](/home/hangjianli/Projects/agentBench/benchmarks/tau2_bench/repo/src/tau2/knowledge/embedders/openai_embedder.py#L71)
- [openai_embedder.py](/home/hangjianli/Projects/agentBench/benchmarks/tau2_bench/repo/src/tau2/knowledge/embedders/openai_embedder.py#L76)

Effect:

- LM Studio can continue using the OpenAI-compatible transport path
- Qwen-family embedding models now get the same query-side formatting semantics that previously only existed in the OpenRouter path

### 3. OpenRouter embedder now reuses the shared helper

Updated:

- [openrouter_embedder.py](/home/hangjianli/Projects/agentBench/benchmarks/tau2_bench/repo/src/tau2/knowledge/embedders/openrouter_embedder.py)

Code changes:

- imports shared query-formatting helpers
- resolves query instruction through shared logic
- formats query text through shared logic
- keeps OpenRouter-specific transport and retry behavior unchanged

Relevant lines:

- [openrouter_embedder.py](/home/hangjianli/Projects/agentBench/benchmarks/tau2_bench/repo/src/tau2/knowledge/embedders/openrouter_embedder.py#L11)
- [openrouter_embedder.py](/home/hangjianli/Projects/agentBench/benchmarks/tau2_bench/repo/src/tau2/knowledge/embedders/openrouter_embedder.py#L98)
- [openrouter_embedder.py](/home/hangjianli/Projects/agentBench/benchmarks/tau2_bench/repo/src/tau2/knowledge/embedders/openrouter_embedder.py#L111)
- [openrouter_embedder.py](/home/hangjianli/Projects/agentBench/benchmarks/tau2_bench/repo/src/tau2/knowledge/embedders/openrouter_embedder.py#L186)

Why:

- avoid duplicating Qwen query-formatting logic in multiple embedder implementations
- keep provider adapters focused on transport/auth/retry behavior

### 4. Document embedding paths explicitly suppress query formatting

Updated:

- [embedding_indexer.py](/home/hangjianli/Projects/agentBench/benchmarks/tau2_bench/repo/src/tau2/knowledge/document_preprocessors/embedding_indexer.py)
- [chunked_embedding_indexer.py](/home/hangjianli/Projects/agentBench/benchmarks/tau2_bench/repo/src/tau2/knowledge/document_preprocessors/chunked_embedding_indexer.py)

Code changes:

- for `openai` and `openrouter` document embedding paths, set `query_instruction=""` when constructing the embedder

Relevant lines:

- [embedding_indexer.py](/home/hangjianli/Projects/agentBench/benchmarks/tau2_bench/repo/src/tau2/knowledge/document_preprocessors/embedding_indexer.py#L58)
- [chunked_embedding_indexer.py](/home/hangjianli/Projects/agentBench/benchmarks/tau2_bench/repo/src/tau2/knowledge/document_preprocessors/chunked_embedding_indexer.py#L75)

Effect:

- document embeddings remain raw
- only query embeddings receive Qwen instruction formatting
- fixes the semantic mismatch without corrupting the document side of the index

### 5. Query encoder docs remain aligned with actual behavior

Updated:

- [embedding_encoder.py](/home/hangjianli/Projects/agentBench/benchmarks/tau2_bench/repo/src/tau2/knowledge/input_preprocessors/embedding_encoder.py)

Code change:

- docstring now states that Qwen formatting is applied for Qwen models generally, rather than only “via OpenRouter”

Relevant lines:

- [embedding_encoder.py](/home/hangjianli/Projects/agentBench/benchmarks/tau2_bench/repo/src/tau2/knowledge/input_preprocessors/embedding_encoder.py#L22)
- [embedding_encoder.py](/home/hangjianli/Projects/agentBench/benchmarks/tau2_bench/repo/src/tau2/knowledge/input_preprocessors/embedding_encoder.py#L66)

Why:

- the behavior is now model-driven and shared across provider paths

### 6. Test coverage updated for the LM Studio / OpenAI-compatible path

Updated:

- [test_retrieval_e2e.py](/home/hangjianli/Projects/agentBench/benchmarks/tau2_bench/repo/tests/test_domains/test_banking_knowledge/test_retrieval_e2e.py)

Added or updated checks for:

- OpenRouter cache config still includes `_query_instruction`
- OpenAI-compatible Qwen cache config now includes `_query_instruction`
- OpenAI-compatible non-Qwen models still do not include `_query_instruction`
- different query instructions still produce different cache configs
- `OpenAIEmbedder` formats Qwen query text before the embedding call
- `EmbeddingIndexer` disables query formatting for document embeddings
- `ChunkedEmbeddingIndexer` disables query formatting for chunked document embeddings

Relevant lines:

- [test_retrieval_e2e.py](/home/hangjianli/Projects/agentBench/benchmarks/tau2_bench/repo/tests/test_domains/test_banking_knowledge/test_retrieval_e2e.py#L1008)
- [test_retrieval_e2e.py](/home/hangjianli/Projects/agentBench/benchmarks/tau2_bench/repo/tests/test_domains/test_banking_knowledge/test_retrieval_e2e.py#L1026)
- [test_retrieval_e2e.py](/home/hangjianli/Projects/agentBench/benchmarks/tau2_bench/repo/tests/test_domains/test_banking_knowledge/test_retrieval_e2e.py#L1038)
- [test_retrieval_e2e.py](/home/hangjianli/Projects/agentBench/benchmarks/tau2_bench/repo/tests/test_domains/test_banking_knowledge/test_retrieval_e2e.py#L1049)
- [test_retrieval_e2e.py](/home/hangjianli/Projects/agentBench/benchmarks/tau2_bench/repo/tests/test_domains/test_banking_knowledge/test_retrieval_e2e.py#L1073)
- [test_retrieval_e2e.py](/home/hangjianli/Projects/agentBench/benchmarks/tau2_bench/repo/tests/test_domains/test_banking_knowledge/test_retrieval_e2e.py#L1104)

Verification command:

```bash
uv run --with pytest python -m pytest benchmarks/tau2_bench/repo/tests/test_domains/test_banking_knowledge/test_retrieval_e2e.py -k 'EncoderCacheConfigIncludesInstruction or OpenAICompatibleQwenFormatting' -q
```

Observed result:

- `7 passed`

## Proposed Design

### Overview

Replace full-document semantic retrieval for banking embedding configs with chunk-level semantic retrieval plus document-level aggregation.

The design has two main components:

1. `chunked_embedding_indexer`
2. `semantic_chunk` retriever

### Component 1: Chunked Embedding Indexer

Responsibilities:

- preserve the original document list in pipeline state
- split each source document into overlapping text chunks
- optionally prefix each chunk with the document title
- embed chunk text instead of full document text
- store chunk embeddings and chunk-to-document mappings in state

Required state:

- `chunk_embeddings`
- `chunk_embeddings_chunk_ids`
- `chunk_embeddings_doc_ids`
- optionally `chunk_embeddings_texts`

Important constraint:

The canonical `documents`, `doc_content_map`, and `doc_title_map` should still refer to original documents, not chunks. `KB_search` returns document-level results today and that contract should remain stable.

### Component 2: Semantic Chunk Retriever

Responsibilities:

- compute cosine similarity between query embedding and chunk embeddings
- score chunks first
- aggregate chunk scores back to original document IDs
- return document-level `(doc_id, score)` pairs

Aggregation policy in the first pass:

- document score = max score over its chunks

This is intentionally simple and easy to reason about. We can later consider:

- max + mean blend
- top-2 average
- category-aware boosts

### Title-Aware Bias

The first pass may include a small boost when the query token set overlaps the document title.

Reason:

- card-product tasks often include exact product names
- generic policy documents can be semantically broad and otherwise outrank the correct product document

Constraint:

- keep the boost small enough that it only breaks near-ties
- do not let lexical title matching dominate semantic evidence

## Why This Design

### Why chunking

Long documents produce diluted embeddings. A chunked index lets the retriever match the part of the document that actually mentions:

- annual fee
- eligibility
- rewards structure
- card-specific terms

instead of embedding the full policy document as one large vector.

### Why aggregate back to documents

The tooling and prompts already assume document-level retrieval. Returning chunks directly would require:

- tool output changes
- prompt changes
- likely evaluation behavior changes

That is too much surface area for the first fix.

### Why not just tune prompts

The task trace shows the agent repeatedly searched but was fed weak retrieval results. Prompting can help the agent query better, but it cannot fix a retriever that ranks the wrong source documents above the right ones.

## Alternatives Considered

### Alternative A: Keep full-document embeddings and just swap the embedding model

Pros:

- minimal implementation work

Cons:

- we already tested a stronger local embedding model path and still saw poor retrieval behavior
- does not address the long-document dilution problem

Decision:

- not sufficient

### Alternative B: Add a reranker only

Pros:

- can improve final ranking

Cons:

- rerankers are only useful if the correct document is already in the candidate set
- current failures suggest the correct candidates are not consistently being retrieved early enough

Decision:

- useful later, but not enough as the first fix

### Alternative C: Hybrid BM25 + semantic retrieval

Pros:

- strong exact-match behavior for product names and fee terms
- often more robust than pure dense retrieval

Cons:

- larger behavioral change
- more moving parts to tune
- may still benefit from chunking on the semantic side

Decision:

- good second-step option if chunked semantic retrieval alone is not enough

### Alternative D: Chunked retrieval plus chunk-level output

Pros:

- potentially better evidence display

Cons:

- changes the `KB_search` tool contract
- wider blast radius

Decision:

- out of scope for the first pass

## Expected Benefits

1. Better recall for card-specific and fee-specific subtopics.
2. Lower rate of generic banking documents outranking the correct product document.
3. Better support for tasks where one document contains many sections and only one section matters.
4. Better alignment between query intent and evidence actually surfaced to the agent.

## Risks

### Risk 1: Chunking does not actually produce useful chunk boundaries

If the chunker merges most content back into one big chunk, there is no retrieval benefit.

Mitigation:

- validate chunk counts on representative documents
- test chunk lengths and boundary behavior directly

### Risk 2: Cache warmup no longer matches runtime indexing

The current prewarm path assumes full-document embeddings. Once runtime uses chunk embeddings, cache warmup must use the same chunking logic or it becomes ineffective and misleading.

Mitigation:

- update warmup to precompute chunk embeddings with the same config as runtime

### Risk 3: Title boost overfits to product-name tasks

If the lexical boost is too strong, it can suppress semantically correct generic policy docs.

Mitigation:

- keep the boost small
- evaluate on multiple task types, not just card recommendation tasks

### Risk 4: Token-based title overlap is too naive

Simple token overlap can misfire on generic words like `card`, `account`, `fee`.

Mitigation:

- consider stopword filtering or product-name extraction in a second pass

## Implementation Plan

### Phase 1: Infrastructure

1. Add `chunked_embedding_indexer`
2. Add `semantic_chunk` retriever
3. Register both in the knowledge registry
4. Wire embedding-based banking retrieval configs to use the new pair

### Phase 2: Correctness

1. Add focused unit tests for:
   - chunk generation
   - state population
   - chunk-to-document aggregation
   - title overlap boost behavior
2. Fix the cache warmup path so it precomputes chunk embeddings, not full-document embeddings

### Phase 3: Evaluation

1. Re-run `task_001`
2. Run a small sample of banking tasks
3. Compare against:
   - `bm25`
   - existing embedding path
   - possible hybrid path if needed

## Evaluation Plan

### Unit-Level

Validate:

- chunk count is greater than 1 for representative long documents
- returned retrieval results stay document-level
- the right document ranks first in synthetic chunked tests

### Task-Level

Start with:

- `task_001`

Then run a small set across categories such as:

- card recommendations
- annual fee policy
- eligibility
- card logistics

### Success Criteria

Minimum:

- retrieval returns relevant product documents more consistently than the current embedding path
- the task-001 recommendation/application failure is fixed

Better:

- measurable gain on a small sampled banking task set versus current `openai_embeddings`

## Open Questions

1. Should the first production variant remain named `openai_embeddings`, or should we introduce a more explicit name like `local_semantic_chunks`?
2. Should we combine BM25 with chunked semantic retrieval now, or only after evaluating chunked semantic retrieval alone?
3. Should chunk text include title prefixes by default, or should that be configurable per variant?
4. Should score aggregation be `max`, `top-2 average`, or a weighted combination?
5. Do we want the cache warmer to store chunk metadata explicitly so runtime does not rebuild chunk docs each run?

## Current Recommendation

Proceed with chunked semantic retrieval as the next implementation step, but do not treat the current patch as ready yet.

The two blockers before calling this design validated are:

1. the chunker must produce real multi-chunk outputs on long documents
2. the cache warmup path must be aligned with chunked runtime indexing
