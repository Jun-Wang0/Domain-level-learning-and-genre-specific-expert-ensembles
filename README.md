# Domain-Level Learning and Genre-Specific Expert Ensembles

Text-only RO3 code accompanying **Integrating Domain-Level Learning and Genre-Specific Expert Ensembles for Cross-Domain Sentiment Analysis**.

The workflow trains a source-domain sentiment model, initializes ten genre-specific sentiment experts from that model, and evaluates on the other two domains. The release includes both source-genre routing and uniform expert averaging. Book, Game, and Movie yield six directed transfer pairs.

This repository contains **code and documentation only**. Review datasets, trained checkpoints, original execution outputs, and experiment result files are not included. No metaphor identification, metaphor features, or feature-fusion model is used in this release.

## Entry Points

| File | Purpose |
| --- | --- |
| `Sentiment_Model_Training.ipynb` | Colab entry point for the shared model and genre experts |
| `main_book_to_light.ipynb` | Full-target-domain evaluation in all six directions, despite the historical filename |
| `Genre_Router_Training.ipynb` | Actual multi-label genre classifier training, found in `Book_Genre_classification.ipynb` |
| `Genre Classification.ipynb` | Archived, optional corpus-analysis notebook. It does **not** train the router |
| `train_sentiment.py` | Shared implementation of the text-only training allocations |
| `train_genre_router.py` | Source-only router training and optional source development split |
| `evaluate_cross_domain.py` | Shared, uniform, and routed evaluation with incremental result saving |

The first three notebooks call the Python modules in this repository, so clone the repository before running them. Their setup cells do this in Colab. Drive originals are not modified.

## Environment

Use Python 3.10 or 3.11 for a new environment. A CUDA GPU is strongly recommended for `roberta-large`. The sentiment defaults use full precision, as in the extracted text-only training code. GPU memory requirements depend on batch size and runtime. Evaluation loads one expert at a time rather than keeping all ten on the GPU.

```bash
git clone https://github.com/Jun-Wang0/Domain-level-learning-and-genre-specific-expert-ensembles.git
cd Domain-level-learning-and-genre-specific-expert-ensembles
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Dependencies are bounded ranges, not a historical environment lockfile. The included tests provide offline checks using synthetic inputs for training, checkpoint loading, ensemble aggregation, metrics, and result saving. Full training has not been rerun as part of preparing this release.

## Data and Configuration

Configure the dataset and output locations for your own environment before running the notebooks. `RO3_ROOT` specifies the corpus root, and the environment variables below allow individual inputs and outputs to be configured.

The sentiment loader reads only review text and the binary sentiment label. Other columns are ignored.

- Sentiment CSVs and genre-specific Excel files need `review` (alternatively `sentence` or `text`) and `sentiment` (alternatively `label`). Labels are `0` for negative and `1` for positive. Singleton-list labels such as `[1]` are accepted. Invalid labels cause an error, not silent conversion to negative.
- Router CSVs need `review` and ten binary genre-indicator columns with the names in `ro3_common.GENRES`. Multiple positive genre labels per review are allowed. The column order is saved with the router.
- `RO3_BOOK_CSV`, `RO3_GAME_CSV`, and `RO3_MOVIE_CSV` override full-domain sentiment paths. `RO3_BOOK_GENRE_CSV`, `RO3_GAME_GENRE_CSV`, and `RO3_MOVIE_GENRE_CSV` override router paths.
- `RO3_OUTPUT` specifies where generated checkpoints and evaluation results are saved. Existing experiment files are not overwritten. Changing this variable selects a separate experiment directory.

Data must be supplied separately with appropriate access rights. This release does not redistribute review text or silently clean, resplit, or deduplicate the experimental corpora.

## Training

```bash
python train_sentiment.py --allocation 4+0
python train_sentiment.py --allocation 2+2
python train_genre_router.py --mode full
```

These commands cover all three source domains. Add `--sources Book` for one source. The four expert allocations are `0+4`, `1+3`, `2+2`, and `3+1`, where the first number denotes source-domain epochs and the second denotes genre-specific epochs. `4+0` trains the single All baseline. Each expert starts independently from the same source-trained checkpoint, not from the preceding expert.

| Setting | Sentiment model | Full-source genre router |
| --- | --- | --- |
| Backbone | RoBERTa-large | RoBERTa-large |
| Maximum input length | 128 tokens | 128 tokens |
| Batch size | 32 | 32 |
| Learning rate | 1e-5 | 3e-5 |
| Epochs | Allocation totals four | Four |
| Optimizer | AdamW | AdamW |
| Weight decay | 0.01 | 0.01 |
| Schedule | Linear, 10% warmup per stage | Linear, 10% warmup |
| Loss | Binary-class cross-entropy | Multi-label BCE with logits |
| Checkpoint selection | Final epoch | Final epoch |

The release fixes seed 42 unless overridden. Historical notebook cells did not consistently set a seed, so fresh runs are not asserted to reproduce historical scores exactly. Domain-level training here means supervised sentiment training on the source corpus, not additional language-model pre-training. Only source data are used for training.

The optional router development mode preserves the original source-only 80/20 split, four epochs, gradient accumulation of two, and selection by validation micro-F1. For the original learning-rate grid, run separately with `--learning-rate 1e-5`, `2e-5`, and `3e-5`. Development checkpoints are saved separately from full-source routers.

```bash
python train_genre_router.py --sources Book --mode development --learning-rate 3e-5
```

The archived `Genre Classification.ipynb` uses a different exploratory 70/10/20 corpus split. It is **not** a required preprocessing step and was not substituted for the main full-source/full-target evaluation protocol.

## Cross-Domain Evaluation

```bash
python evaluate_cross_domain.py --allocation 2+2 --aggregation both
```

By default, this evaluates All (4+0), GB-Uniform (2+2), and GB-Routed (2+2) on the full target corpus for Book to Game, Book to Movie, Game to Book, Game to Movie, Movie to Book, and Movie to Game. It does not split the target corpus into genre-specific subsets. Use `--aggregation uniform` to evaluate without a router, or `--skip-baseline` to omit All.

Each expert produces two raw sentiment logits `z_k(x)`. Uniform aggregation computes `sum_k z_k(x) / 10`. Routed aggregation computes `sum_k softmax(r(x))_k * z_k(x)`, where `r(x)` is the source genre router's ten-dimensional logit vector. The predicted class is the argmax of the combined logits. This is **not** probability averaging and does not select a single expert.

The router is trained as a multi-label classifier but its raw outputs are normalized with softmax for expert mixing at evaluation time, following the source notebook. These weights are mixing coefficients, not calibrated cross-domain genre probabilities. A confident routing weight does not establish that the associated sentiment expert is correct.

Accuracy and macro-averaged precision, recall, and F1 are reported in percent. All ten experts are required. Results and prediction labels, without review text, are saved after each completed pair in a new timestamped results directory. Missing or incompatible checkpoints cause an error rather than silent expert omission.

### Existing Checkpoints

To evaluate existing models instead of newly trained ones, configure `RO3_SENTIMENT_MODELS` and `RO3_ROUTER_MODELS` to identify the corresponding checkpoint locations before running the evaluator.

Router outputs must match expert names. New routers save semantic `id2label` mappings. Old routers with generic names such as `LABEL_0` require a JSON mapping supplied through `--router-labels`. This mapping lists the ten genre names for each source domain in the **original training CSV column order**. Do not infer that order from the alphabetical expert list. Uniform averaging does not require this mapping. The release cannot retroactively certify the correspondence between every historical checkpoint and result table.

## Scope

This is a cleaned and consolidated code release, not an archive of completed runs. It does not include Roy/IMO comparison implementations, experimental top-k variants, older BERT prototypes, or trained models. `SOURCE_PROVENANCE.json` records the source notebook names and hashes. No new results or claims of exact historical reproduction are made by publishing the code.
