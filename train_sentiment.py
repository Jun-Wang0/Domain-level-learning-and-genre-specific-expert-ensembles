"""Source-domain learning and independent genre-specific sentiment fine-tuning."""

import argparse
import copy
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (AutoTokenizer, RobertaConfig, RobertaForSequenceClassification,
                          get_linear_schedule_with_warmup, set_seed)

from common import (ALLOCATIONS, GENRES, MAX_LENGTH, MODEL_NAME, clear_memory,
                        domain_csv, genre_subset, load_reviews, sentiment_path, write_json)


class ReviewDataset(Dataset):
    def __init__(self, path, tokenizer):
        self.texts, self.labels = load_reviews(path)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        encoded = self.tokenizer(self.texts[index], max_length=MAX_LENGTH,
                                 padding="max_length", truncation=True, return_tensors="pt")
        return {"input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "labels": torch.tensor(self.labels[index], dtype=torch.long)}


def train_stage(model, loader, epochs, learning_rate, device):
    if epochs == 0:
        return model
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * steps), num_training_steps=steps)
    model.to(device).train()
    for epoch in range(epochs):
        progress = tqdm(loader, desc="Epoch %d/%d" % (epoch + 1, epochs))
        for batch in progress:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite sentiment training loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            progress.set_postfix(loss=loss.item())
    return model


def save_checkpoint(model, tokenizer, path, metadata):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    write_json(path / "training_metadata.json", metadata)


def train_source(domain, allocation, batch_size=32, learning_rate=1e-5, seed=42,
                 base_model=MODEL_NAME, device=None):
    domain_epochs, genre_epochs = ALLOCATIONS[allocation]
    destinations = ([sentiment_path(domain, "4+0")] if allocation == "4+0" else
                    [sentiment_path(domain, allocation, genre) for genre in GENRES[domain]])
    if any(path.exists() for path in destinations):
        raise FileExistsError("A destination already exists. Use a fresh ENSEMBLE_OUTPUT; "
                              "this release does not overwrite or silently resume experiments.")
    inputs = ([domain_csv(domain)] if domain_epochs else [])
    if genre_epochs:
        inputs += [genre_subset(domain, genre) for genre in GENRES[domain]]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)

    set_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True, add_prefix_space=True)
    config = RobertaConfig.from_pretrained(base_model, num_labels=2,
                                          id2label={0: "negative", 1: "positive"},
                                          label2id={"negative": 0, "positive": 1})
    metadata = {"domain": domain, "allocation": allocation, "seed": seed,
                "base_model": base_model, "batch_size": batch_size,
                "learning_rate": learning_rate, "max_length": MAX_LENGTH,
                "weight_decay": 0.01, "warmup_ratio": 0.1,
                "checkpoint_rule": "final epoch", "torch": torch.__version__}
    shared_state = None
    if domain_epochs:
        dataset = ReviewDataset(domain_csv(domain), tokenizer)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model = RobertaForSequenceClassification.from_pretrained(base_model, config=config)
        train_stage(model, loader, domain_epochs, learning_rate, device)
        if allocation == "4+0":
            save_checkpoint(model, tokenizer, destinations[0], metadata)
        else:
            model.cpu()
            shared_state = copy.deepcopy(model.state_dict())
        del model, loader, dataset
        clear_memory()

    for genre in GENRES[domain] if genre_epochs else []:
        if shared_state is None:
            model = RobertaForSequenceClassification.from_pretrained(base_model, config=config)
        else:
            model = RobertaForSequenceClassification(config)
            model.load_state_dict(shared_state, strict=True)
        dataset = ReviewDataset(genre_subset(domain, genre), tokenizer)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        train_stage(model, loader, genre_epochs, learning_rate, device)
        save_checkpoint(model, tokenizer, sentiment_path(domain, allocation, genre),
                        dict(metadata, genre=genre))
        del model, loader, dataset
        clear_memory()
    del shared_state
    clear_memory()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", choices=list(GENRES), default=list(GENRES))
    parser.add_argument("--allocation", choices=list(ALLOCATIONS), default="2+2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    for source in args.sources:
        train_source(source, args.allocation, args.batch_size, args.learning_rate, args.seed)


if __name__ == "__main__":
    main()
