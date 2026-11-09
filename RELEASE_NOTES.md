# Release Notes

## Source and Scope

This code-only release was prepared from the author's Google Drive notebooks. `SOURCE_PROVENANCE.json` records SHA-256 hashes of the downloaded originals for traceability. Original notebooks, including their outputs, remain outside the public release.

- `Sentiment_Model_Training.ipynb` supplied the standard RoBERTa sentiment branch and the domain/genre epoch allocations. Repeated cells have been consolidated into `train_sentiment.py` and a Colab entry point. The related `Sentiment_Model_Training2.ipynb` and `Sentiment_Model_Training3.ipynb` were inspected but are not included as redundant copies.
- `main_book_to_light.ipynb` supplied full-target evaluation and raw-logit aggregation. Its separate uniform and routed runs are exposed through one evaluator for all six directions. Metaphor-feature parsing, projection/fusion classes, metaphor-enabled loops, top-k prototypes, and older unrelated model experiments have been removed from the publication version.
- The requested `Genre Classification.ipynb` contained exploratory splitting and corpus statistics, not model training. Its source is retained as an explicitly optional notebook. The splitting cell is placed before analysis, generated files are redirected to a separate exploratory output directory, and execution requires opt-in.
- The additional `Book_Genre_classification.ipynb` supplied the actual multi-label router training and source-only development procedure. The extracted implementation is `train_genre_router.py`, with `Genre_Router_Training.ipynb` as its Colab entry point.

No datasets, checkpoint weights, cached representations, prediction files, paper result tables, or credentials are published. No new model comparison experiment was run during release preparation.

## Deliberate Changes

The new modules preserve the main text-only architecture, epoch allocations, losses, and aggregation formulas. The following operational changes are explicit rather than hidden in a claim of exact reproduction.

1. Paths are configurable, and default outputs use a new directory instead of overwriting historical checkpoints.
2. The release exposes all three domains and the four expert allocations through configuration rather than commented task lists and duplicated cells.
3. A fixed seed of 42 is set explicitly. Original sentiment cells did not consistently fix seeds.
4. Invalid labels, missing experts, incomplete checkpoints, and non-finite outputs raise errors. Some original evaluation cells instead substituted negative labels or skipped unavailable experts.
5. Router column order is saved and checked. Historical generic label names do not prove the mapping from router channels to genre experts, so evaluation requires a verified order for such checkpoints.
6. Evaluation loads one expert at a time and combines its raw logits on the CPU. The mathematical aggregation is retained, but bitwise equivalence across hardware and kernels is not asserted.
7. Results are persisted after each completed transfer pair, with explicit macro metric names and units. All notebook execution outputs are removed.

Sentiment training uses the full supplied source corpus and the final epoch checkpoint. Router `full` mode likewise uses the supplied full source genre corpus. Optional router development and the archived exploratory corpus split are separate procedures. Publishing this code does not establish the absence of overlap in supplied corpora or validate every historical checkpoint-to-result association.

## Verification

All 13 offline tests passed during release preparation. They check label parsing, macro metrics, expert weighting, router-channel reordering, all six evaluation directions, result saving, and notebook structure. A small randomly initialized RoBERTa model is used to test actual sentiment training and multi-label router training, checkpoint saving, local reloading, and prediction. These tests use synthetic inputs, not the research datasets, and are not benchmark results.

Verification environment used during release preparation was Windows, Python 3.9, PyTorch 2.0.1, Transformers 4.40.2, Datasets 2.19.1, Accelerate 0.30.1, NumPy 1.24.1, pandas 2.0.2, scikit-learn 1.2.2, and nbformat 5.10.4. These are recorded as the local check environment, not as a recommended historical lockfile. For a new environment, use Python 3.10 or 3.11 and `requirements.txt`.

Full RoBERTa-large training, full-dataset evaluation, and the optional exploratory analysis were not executed during preparation. Reproducing historical numerical results also requires the matching data, preprocessing, checkpoint provenance, seeds, and runtime environment. The repository alone does not provide those artifacts.
