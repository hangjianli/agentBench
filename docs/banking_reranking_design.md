# Banking Knowledge Reranking Design

## Status

Draft

## Problem

The current `banking_knowledge` retrieval stack is now chunk-aware at the
embedding stage, but the ranking quality is still weak for broad product
queries.

Observed behavior from the latest `task_001` smoke test:

1. the task succeeded end-to-end
2. early `KB_search` calls were still noisy
3. broad queries returned generic rewards docs, business-card docs, and
   account-benefit docs above the most useful personal-card docs
4. the agent recovered only after issuing narrower follow-up queries

This means chunked retrieval improved recall, but first-pass ranking is still
not precise enough.

## Current Interface Constraints

The design should fit the current retrieval interfaces instead of replacing
them.

### Existing extension points

1. `RetrievalPipeline` already supports retrievers plus postprocessors in
   sequence.
2. `BaseRetriever.retrieve(input_data, state)` returns
   `List[Tuple[doc_id, score]]`.
3. `BasePostprocessor.process(results, input_data, state)` can reorder or
   filter those results.
4. `PipelineSpec` in
   `benchmarks/tau2_bench/repo/src/tau2/domains/banking_knowledge/retrieval.py`
   already has a `reranker` flag and `reranker_min_score`.

### Current limitations

1. the current `reranker` flag is only boolean, so the variant cannot choose
   reranker type, model, or candidate-pool size
2. `_create_kb_pipeline(...)` always wires reranking to
   `pointwise_llm_reranker`
3. `pointwise_llm_reranker` defaults to `gpt-5.2`, which does not match the
   local LM Studio setup using `AGENT_BACKBONE_MODEL`
4. the reranker currently sees canonical full-document text from
   `state["doc_content_map"]`, not the chunk(s) that actually matched the query
5. the reranker only receives the retriever's top-`k`, so it cannot rescue a
   useful document that ranked just outside that cutoff

## Why The Current Reranker Path Is Not Enough

Even though the codebase already has reranker variants such as
`openai_embeddings_reranker`, the current implementation is not a strong fit
for the LM Studio-backed banking setup.

### Issue 1: wrong default reranker model

`pointwise_llm_reranker` currently defaults to `gpt-5.2`, while the local
runtime uses:

- `AGENT_BACKBONE_MODEL=qwen/qwen3.5-9b`
- `OPENAI_BASE_URL=http://localhost:1234/v1`

Without an explicit reranker model override, the reranker path depends on a
model name that the local LM Studio server may not host.

### Issue 2: reranking happens too late on too few candidates

The initial semantic retriever returns only the final top-`k` documents.
Reranking those same `k` results can reorder them, but cannot recover a good
document that finished at rank 11 or 12.

### Issue 3: reranker judges the wrong text unit

Retrieval is chunk-based, but reranking currently reads the full document body
from `doc_content_map`. That dilutes the signal for long, multi-topic banking
documents.

For product search, the reranker should judge the best matching chunk evidence,
or at least a small set of top chunks, instead of the entire document.

## Goals

1. improve first-pass document ranking for broad product queries
2. keep the external `KB_search` tool contract unchanged
3. preserve the current `BaseRetriever` and `BasePostprocessor` interfaces
4. make reranking work with the local LM Studio model configuration
5. keep the implementation incremental and testable

## Non-Goals

1. replacing the retrieval pipeline API
2. introducing a separate external reranking service
3. changing the user-visible `KB_search` output format in the first pass
4. building a banking-domain-specific rules engine as the only ranker

## Proposed Design

Use a two-stage ranking architecture that stays inside the current pipeline
interfaces:

1. semantic chunk retrieval produces a larger candidate set
2. a reranker postprocessor reranks those candidates using chunk-level evidence
3. the postprocessor returns the final top-`k` document IDs

### Stage 1: widen the candidate pool

Add an internal distinction between:

- `candidate_k`: how many documents the retriever should surface for reranking
- `top_k`: how many documents should be returned after reranking

Recommendation:

- default `candidate_k = 25`
- default final `top_k = 10`

This gives the reranker room to rescue good documents that are slightly
under-ranked by the embedding retriever.

### Stage 2: rerank with chunk evidence

Teach `SemanticChunkRetriever` to attach chunk evidence for the current query
to `input_data`, while still returning the same `(doc_id, score)` tuples.

Recommended transient structure:

```python
input_data["_retrieval_context"] = {
    "semantic_chunk": {
        "doc_candidates": {
            doc_id: {
                "initial_score": float,
                "top_chunks": [
                    {
                        "chunk_id": str,
                        "chunk_text": str,
                        "chunk_score": float,
                    }
                ],
            }
        }
    }
}
```

Why this fits the current API:

1. retrievers already receive the mutable `input_data` dict
2. postprocessors already receive the same `input_data` dict
3. no retriever or postprocessor base-interface changes are required

### Stage 3: rerank the evidence, not the whole document

Introduce a banking-oriented reranker postprocessor that prefers chunk evidence
over full-document text.

Recommended postprocessor name:

- `chunk_evidence_llm_reranker`

Behavior:

1. read the candidate docs from `results`
2. read top chunk evidence from `input_data["_retrieval_context"]`
3. build a compact reranker passage per candidate using:
   - document title
   - best chunk
   - optionally second-best chunk if it adds distinct evidence
4. ask the reranker model to score relevance for the current user query
5. sort candidates by reranker score
6. return the final top-`k`

This keeps the reranker focused on the text that actually caused the retrieval
match.

## Configuration Changes

Replace the boolean reranker flag in `PipelineSpec` with a richer optional
specification.

### Proposed shape

```python
@dataclass
class RerankerSpec:
    type: Literal["pointwise_llm", "chunk_evidence_llm"]
    model: Optional[str] = None
    min_score: Optional[float] = None
    top_k: Optional[int] = None
    candidate_k: int = 25
    max_chunks_per_doc: int = 2


@dataclass
class PipelineSpec:
    type: Literal["embedding", "bm25"]
    top_k: int = 10
    embedder_type: Optional[str] = None
    embedder_model: Optional[str] = None
    reranker: Optional[RerankerSpec] = None
```

Why this is better:

1. variants can choose reranker type explicitly
2. variants can pass the correct local model
3. retriever candidate depth becomes configurable
4. existing variant names can stay the same

## Recommended Default For LM Studio

For the current local setup, the reranker model should default to:

- `os.getenv("RERANKER_MODEL")` if set
- otherwise `os.getenv("AGENT_BACKBONE_MODEL")`

This aligns reranking with the locally hosted LM Studio chat model instead of
hardcoding `gpt-5.2`.

## Implementation Plan

### Phase 1: make reranker config explicit

1. add `RerankerSpec`
2. update banking retrieval variants to use structured reranker config
3. default reranker model to `RERANKER_MODEL` or `AGENT_BACKBONE_MODEL`

### Phase 2: separate candidate depth from final top-k

1. pass `candidate_k` to the retriever
2. keep postprocessor `top_k` as the final returned count
3. verify that `KB_search` output size remains unchanged

### Phase 3: expose semantic chunk evidence

1. extend `ChunkedEmbeddingIndexer` state with direct chunk text lookup if
   needed
2. update `SemanticChunkRetriever` to collect top chunk hits per document
3. stash that evidence in `input_data["_retrieval_context"]`

### Phase 4: add chunk-evidence reranker

1. implement `chunk_evidence_llm_reranker`
2. build reranker prompts from title plus top chunk evidence
3. run on the candidate pool and output final reranked docs

### Phase 5: evaluate

1. compare `openai_embeddings` vs `openai_embeddings_reranker`
2. inspect first-turn `KB_search` rankings on `task_001`
3. verify whether required card docs move into the top 3 more consistently

## Prompting Strategy For The Reranker

The reranker prompt should be narrower than the current generic pointwise
prompt.

Recommended judging criteria:

1. prefer personal-card documents over business-card documents when the query
   asks about personal cards
2. prefer documents that directly answer annual-fee, eligibility, and rewards
   questions
3. prefer exact product matches over generic rewards-policy docs
4. penalize account-benefit docs unless the query explicitly asks about linked
   accounts or account perks

This can still be model-agnostic, but it should reflect the common error modes
seen in banking retrieval.

## Testing Plan

### Unit tests

1. `SemanticChunkRetriever` records top chunk evidence into `input_data`
2. reranker config resolves local model name correctly
3. candidate pool depth is larger than final output depth
4. chunk-evidence reranker prefers personal-card evidence over business-card
   evidence for a personal-card query

### Integration tests

1. run retrieval-only checks on `task_001` query variants
2. assert `doc_credit_cards_gold_rewards_card_001`,
   `doc_credit_cards_silver_rewards_card_001`,
   `doc_credit_cards_bronze_rewards_card_001`, and
   `doc_credit_cards_platinum_rewards_card_001` appear near the top for the
   relevant recommendation queries

### Smoke test

1. rerun the 1-task banking smoke test with reranking enabled
2. compare:
   - first-turn retrieved docs
   - total reward
   - timing overhead

## Alternatives Considered

### 1. Just enable the existing `pointwise_llm_reranker`

Rejected as the sole fix because:

1. it currently assumes the wrong default model for LM Studio
2. it reranks too few candidates
3. it judges full documents instead of matched chunks

### 2. Put all reranking logic inside `SemanticChunkRetriever`

Rejected because:

1. it would blur retrieval and postprocessing responsibilities
2. it would bypass the existing postprocessor extension point
3. it would make BM25 and future retrievers harder to reuse

### 3. Add a domain-specific hardcoded scoring heuristic only

Rejected as the primary plan because:

1. it would overfit to the current failure mode
2. it would not generalize well to broader banking queries
3. it is still useful as a future lightweight pre-rerank, but not as the main
   design

## Recommendation

Implement reranking as a structured postprocessing layer, not as a retriever
rewrite.

The recommended first implementation is:

1. introduce `RerankerSpec`
2. widen semantic retrieval to `candidate_k`
3. expose top chunk evidence through `input_data`
4. add `chunk_evidence_llm_reranker`
5. default reranker model to `AGENT_BACKBONE_MODEL` for LM Studio compatibility

This is the smallest design that:

1. fits the current interfaces
2. works with the local LM Studio setup
3. addresses the actual failure mode from the smoke test
4. leaves room for cheaper heuristic reranking later if needed
