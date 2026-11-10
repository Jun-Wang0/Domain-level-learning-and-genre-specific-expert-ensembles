"""Multi-label source-domain genre router training."""

import argparse
import inspect

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)

from common import (GENRES, MAX_LENGTH, MODEL_NAME, clear_memory, genre_csv,
                        router_path, write_json)


def prepare_frame(frame, domain):
    order = [column for column in frame.columns if column in GENRES[domain]]
    if "review" not in frame or frame.empty or set(order) != set(GENRES[domain]):
        raise ValueError("Expected review and all ten source-genre columns.")
    values = frame[order].to_numpy(dtype=float)
    if not np.isin(values, [0, 1]).all():
        raise ValueError("Genre targets must be binary multi-hot indicators.")
    if frame["review"].isna().any():
        raise ValueError("Missing router text. Check the input corpus before training.")
    return {"review": frame["review"].astype(str).tolist(),
            "labels": values.tolist()}, order


def router_metrics(prediction):
    # sigmoid(logit) >= 0.5 is equivalent to logit >= 0.
    predicted = (prediction.predictions >= 0).astype(int)
    labels = prediction.label_ids
    return {"micro_f1": f1_score(labels, predicted, average="micro", zero_division=0),
            "macro_f1": f1_score(labels, predicted, average="macro", zero_division=0),
            "accuracy": accuracy_score(labels, predicted)}


def train_router(domain, mode="full", learning_rate=3e-5, seed=42):
    destination = router_path(domain)
    if mode == "development":
        destination = destination.parent / (destination.name + "_dev_" + str(learning_rate))
    if destination.exists():
        raise FileExistsError("Output already exists. Select a fresh ENSEMBLE_OUTPUT.")
    data, order = prepare_frame(pd.read_csv(genre_csv(domain)), domain)
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = Dataset.from_dict(data)
    if mode == "development":
        split = dataset.train_test_split(test_size=0.2, seed=seed)
        train_data, dev_data = split["train"], split["test"]
    else:
        train_data, dev_data = dataset, None

    def tokenize(batch):
        return tokenizer(batch["review"], truncation=True, max_length=MAX_LENGTH)

    train_data = train_data.map(tokenize, batched=True, remove_columns=["review"])
    if dev_data is not None:
        dev_data = dev_data.map(tokenize, batched=True, remove_columns=["review"])
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(order), problem_type="multi_label_classification",
        id2label=dict(enumerate(order)), label2id={label: i for i, label in enumerate(order)})
    options = dict(
        output_dir=str(destination), learning_rate=learning_rate,
        per_device_train_batch_size=32, per_device_eval_batch_size=32,
        gradient_accumulation_steps=2 if mode == "development" else 1,
        num_train_epochs=4, weight_decay=0.01, warmup_ratio=0.1,
        lr_scheduler_type="linear", optim="adamw_torch", logging_steps=50,
        fp16=torch.cuda.is_available(), report_to="none", seed=seed, data_seed=seed,
        save_strategy="epoch" if mode == "development" else "no")
    if dev_data is not None:
        arguments = inspect.signature(TrainingArguments).parameters
        evaluation_key = "eval_strategy" if "eval_strategy" in arguments else "evaluation_strategy"
        options.update({evaluation_key: "epoch", "load_best_model_at_end": True,
                        "metric_for_best_model": "micro_f1", "greater_is_better": True,
                        "save_total_limit": 1})
    trainer_options = dict(model=model, args=TrainingArguments(**options),
                           train_dataset=train_data, eval_dataset=dev_data,
                           data_collator=DataCollatorWithPadding(tokenizer),
                           compute_metrics=router_metrics if dev_data is not None else None)
    tokenizer_key = ("processing_class" if "processing_class" in inspect.signature(Trainer).parameters
                     else "tokenizer")
    trainer_options[tokenizer_key] = tokenizer
    trainer = Trainer(**trainer_options)
    trainer.train()
    trainer.save_model(str(destination))
    tokenizer.save_pretrained(destination)
    write_json(destination / "genre_order.json", order)
    write_json(destination / "training_metadata.json", {
        "domain": domain, "mode": mode, "seed": seed, "learning_rate": learning_rate,
        "epochs": 4, "max_length": MAX_LENGTH, "batch_size": 32,
        "gradient_accumulation_steps": options["gradient_accumulation_steps"],
        "loss": "BCEWithLogitsLoss", "weight_decay": 0.01, "warmup_ratio": 0.1,
        "label_order": order, "training_rows": len(train_data),
        "validation_rows": len(dev_data) if dev_data is not None else 0,
        "checkpoint_rule": "best source validation micro-F1" if dev_data is not None else "final epoch"})
    del trainer, model, dataset, train_data, dev_data
    clear_memory()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", choices=list(GENRES), default=list(GENRES))
    parser.add_argument("--mode", choices=["full", "development"], default="full")
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    for domain in args.sources:
        train_router(domain, args.mode, args.learning_rate, args.seed)


if __name__ == "__main__":
    main()
