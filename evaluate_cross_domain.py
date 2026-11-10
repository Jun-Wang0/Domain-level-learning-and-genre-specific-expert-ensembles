"""Evaluate full target domains using shared, uniformly averaged, and routed models."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from common import (GENRES, MAX_LENGTH, clear_memory, combine_logits, domain_csv,
                        load_reviews, output_root, resolve_genre_order, router_path,
                        sentiment_metrics, sentiment_path, write_json)


def load_local_model(path, device):
    if not (Path(path) / "config.json").is_file():
        raise FileNotFoundError("Missing trained checkpoint: %s" % path)
    model, info = AutoModelForSequenceClassification.from_pretrained(
        path, local_files_only=True, output_loading_info=True)
    # Never evaluate an accidentally reinitialized classification head.
    if info.get("missing_keys") or info.get("mismatched_keys") or info.get("error_msgs"):
        raise ValueError("Incomplete checkpoint %s: %s" % (path, info))
    if info.get("unexpected_keys"):
        raise ValueError("Unexpected checkpoint architecture %s: %s" % (path, info))
    return model.to(device).eval()


def predict_logits(path, texts, batch_size, device, router=False):
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = load_local_model(path, device)
    config = model.config
    if not router and config.num_labels != 2:
        raise ValueError("Sentiment checkpoints must have two output classes.")
    batches = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(texts), batch_size), desc=Path(path).name):
            encoded = tokenizer(texts[start:start + batch_size], padding="max_length",
                                 truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
            logits = model(**{key: value.to(device) for key, value in encoded.items()}).logits
            batches.append(logits.float().cpu())
    del model
    clear_memory()
    result = torch.cat(batches)
    if not torch.isfinite(result).all():
        raise ValueError("Non-finite model outputs from %s" % path)
    return result, config


def ordered_router_weights(logits, config, domain, explicit=None):
    order = resolve_genre_order(config, GENRES[domain], explicit)
    columns = [order.index(genre) for genre in GENRES[domain]]
    return torch.softmax(logits, dim=1)[:, columns]


def evaluate(sources=None, targets=None, allocation="2+2", aggregation="both",
             batch_size=32, router_labels=None, skip_baseline=False, results_dir=None):
    sources = sources or list(GENRES)
    targets = targets or list(GENRES)
    pairs = [(source, target) for source in sources for target in targets if source != target]
    if not pairs:
        raise ValueError("No cross-domain transfer pairs selected.")
    if batch_size < 1:
        raise ValueError("Batch size must be positive.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    explicit_orders = json.loads(Path(router_labels).read_text(encoding="utf-8")) if router_labels else {}
    # Fail before a long run if any required checkpoint or corpus is missing.
    for source, target in pairs:
        if not domain_csv(target).is_file():
            raise FileNotFoundError(domain_csv(target))
        checkpoints = [sentiment_path(source, allocation, genre) for genre in GENRES[source]]
        if not skip_baseline:
            checkpoints.append(sentiment_path(source, "4+0"))
        if aggregation in ("routed", "both"):
            checkpoints.append(router_path(source))
        for checkpoint in checkpoints:
            if not (checkpoint / "config.json").is_file():
                raise FileNotFoundError(checkpoint)
        if aggregation in ("routed", "both"):
            resolve_genre_order(AutoConfig.from_pretrained(router_path(source), local_files_only=True),
                                GENRES[source], explicit_orders.get(source))

    run = Path(results_dir) if results_dir else output_root() / "results" / datetime.now(
        timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "run_config.json", {
        "pairs": pairs, "allocation": allocation, "aggregation": aggregation,
        "batch_size": batch_size, "max_length": MAX_LENGTH, "device": str(device),
        "expert_order": {source: GENRES[source] for source in sources},
        "explicit_router_orders": explicit_orders, "skip_baseline": skip_baseline,
        "metric_average": "macro", "units": "percent", "aggregation_space": "raw logits",
        "target_corpora": {target: str(domain_csv(target)) for target in targets}})
    rows = []
    completed = []
    for source, target in pairs:
        print("\n%s -> %s" % (source, target), flush=True)
        texts, labels = load_reviews(domain_csv(target))
        predictions = {}
        if not skip_baseline:
            logits, _ = predict_logits(sentiment_path(source, "4+0"), texts, batch_size, device)
            predictions["All (4+0)"] = logits.argmax(1).numpy()
        expert_logits = []
        for genre in GENRES[source]:
            logits, _ = predict_logits(sentiment_path(source, allocation, genre), texts, batch_size, device)
            expert_logits.append(logits)
        stacked = torch.stack(expert_logits, dim=1)
        if aggregation in ("uniform", "both"):
            predictions["GB-Uniform (" + allocation + ")"] = combine_logits(stacked).argmax(1).numpy()
        if aggregation in ("routed", "both"):
            logits, config = predict_logits(router_path(source), texts, batch_size, device, router=True)
            weights = ordered_router_weights(logits, config, source, explicit_orders.get(source))
            predictions["GB-Routed (" + allocation + ")"] = combine_logits(stacked, weights).argmax(1).numpy()
        for method, predicted in predictions.items():
            row = {"Source": source, "Target": target, "Method": method, "Reviews": len(texts),
                   **sentiment_metrics(labels, predicted)}
            rows.append(row)
            print(row, flush=True)
        # Persist after each completed pair. Review text is never included in these files.
        pd.DataFrame(rows).to_csv(run / "cross_domain_results.csv", index=False)
        pd.DataFrame({"label": labels, **predictions}).to_csv(
            run / (source + "_to_" + target + "_predictions.csv"), index=False)
        completed.append([source, target])
        write_json(run / "progress.json", {"completed_pairs": completed, "total_pairs": len(pairs)})
        del expert_logits, stacked, predictions, texts, labels, logits
        clear_memory()
    print("Saved results to", run)
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", choices=list(GENRES), default=list(GENRES))
    parser.add_argument("--targets", nargs="+", choices=list(GENRES), default=list(GENRES))
    parser.add_argument("--allocation", choices=["0+4", "1+3", "2+2", "3+1"], default="2+2")
    parser.add_argument("--aggregation", choices=["uniform", "routed", "both"], default="both")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--router-labels", help="JSON mapping source domains to verified router label orders")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--results-dir", help="New output directory; an existing directory is not overwritten")
    args = parser.parse_args(argv)
    evaluate(**vars(args))


if __name__ == "__main__":
    main()
