"""Offline checks with synthetic inputs only. No review corpus or download is needed."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from tokenizers import ByteLevelBPETokenizer
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaModel, RobertaTokenizerFast

import evaluate_cross_domain as evaluation
import train_genre_router as router_training
from ro3_common import (GENRES, binary_label, combine_logits, load_reviews,
                        resolve_genre_order, router_path, sentiment_metrics, sentiment_path)
from train_genre_router import prepare_frame
from train_sentiment import train_stage


class CoreTests(unittest.TestCase):
    def test_binary_labels(self):
        for value in (1, 1.0, "1", "[1]", [1], "['1']"):
            self.assertEqual(binary_label(value), 1)
        for value in (2, "missing", "[]", "[0, 1]", float("nan")):
            with self.assertRaises(ValueError):
                binary_label(value)

    def test_text_loader_does_not_require_auxiliary_features(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviews.csv"
            pd.DataFrame({"review": ["synthetic positive", "synthetic negative"],
                          "sentiment": [1, 0]}).to_csv(path, index=False)
            texts, labels = load_reviews(path)
            self.assertEqual(len(texts), 2)
            self.assertEqual(labels, [1, 0])

    def test_uniform_equals_constant_router_weights(self):
        logits = torch.arange(40, dtype=torch.float32).reshape(2, 10, 2)
        weights = torch.ones(2, 10) / 10
        torch.testing.assert_close(combine_logits(logits), combine_logits(logits, weights))
        torch.testing.assert_close(combine_logits(logits), logits.mean(1))

    def test_aggregation_is_not_probability_averaging(self):
        logits = torch.tensor([[[0., 10.], [0., -2.], [0., -2.]]])
        self.assertEqual(combine_logits(logits).argmax(1).item(), 1)
        self.assertEqual(logits.softmax(-1).mean(1).argmax(1).item(), 0)

    def test_invalid_weights_fail(self):
        with self.assertRaises(ValueError):
            combine_logits(torch.zeros(1, 10, 2), torch.ones(1, 10))
        with self.assertRaises(ValueError):
            combine_logits(torch.full((1, 10, 2), float("nan")))

    def test_router_order_is_reconciled(self):
        order = list(reversed(GENRES["Book"]))
        config = RobertaConfig(num_labels=10, id2label=dict(enumerate(order)))
        logits = torch.arange(10, dtype=torch.float32).reshape(1, 10)
        actual = evaluation.ordered_router_weights(logits, config, "Book")
        torch.testing.assert_close(actual, logits.softmax(1).flip(1))

    def test_unverified_router_order_fails(self):
        config = RobertaConfig(num_labels=10)
        with self.assertRaises(ValueError):
            resolve_genre_order(config, GENRES["Book"])
        self.assertEqual(resolve_genre_order(config, GENRES["Book"], GENRES["Book"]), GENRES["Book"])

    def test_router_training_order_comes_from_columns(self):
        order = list(reversed(GENRES["Book"]))
        frame = pd.DataFrame({"review": ["synthetic"], **{g: [1] for g in order}, "sentiment": [0]})
        data, actual = prepare_frame(frame, "Book")
        self.assertEqual(actual, order)
        self.assertEqual(len(data["labels"][0]), 10)

    def test_macro_recall_for_single_class_predictions(self):
        result = sentiment_metrics([0, 0, 1, 1], [1, 1, 1, 1])
        self.assertEqual(result["Recall"], 50)
        self.assertAlmostEqual(result["F1"], 100 / 3)

    def test_six_directions_and_result_saving(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                "RO3_ROOT": directory, "RO3_OUTPUT": directory + "/out",
                "RO3_SENTIMENT_MODELS": directory + "/sentiment",
                "RO3_ROUTER_MODELS": directory + "/routers"}):
            for domain in GENRES:
                path = Path(directory) / (domain + ".csv")
                pd.DataFrame({"review": ["a", "b"], "sentiment": [0, 1]}).to_csv(path, index=False)
                self.addCleanup(os.environ.pop, "RO3_" + domain.upper() + "_CSV", None)
                os.environ["RO3_" + domain.upper() + "_CSV"] = str(path)
                paths = [sentiment_path(domain, "4+0"), router_path(domain)]
                paths += [sentiment_path(domain, "2+2", g) for g in GENRES[domain]]
                for checkpoint in paths:
                    checkpoint.mkdir(parents=True)
                    (checkpoint / "config.json").write_text("{}", encoding="utf-8")

            def config_for(path, **kwargs):
                domain = Path(path).name.split("_Genre_")[0]
                return RobertaConfig(num_labels=10, id2label=dict(enumerate(GENRES[domain])))

            def fake_predict(path, texts, batch_size, device, router=False):
                if router:
                    return torch.zeros(2, 10), config_for(path)
                return torch.tensor([[2., 0.], [0., 2.]]), RobertaConfig(num_labels=2)

            with patch.object(evaluation.AutoConfig, "from_pretrained", side_effect=config_for), \
                    patch.object(evaluation, "predict_logits", side_effect=fake_predict):
                results = evaluation.evaluate(results_dir=Path(directory) / "results")
            self.assertEqual(len(results), 18)
            self.assertEqual(len(results[["Source", "Target"]].drop_duplicates()), 6)
            self.assertTrue((results["F1"] == 100).all())
            self.assertTrue((Path(directory) / "results/cross_domain_results.csv").is_file())
            progress = json.loads((Path(directory) / "results/progress.json").read_text())
            self.assertEqual(len(progress["completed_pairs"]), 6)


class TinyModelTests(unittest.TestCase):
    def test_offline_train_save_reload_predict(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            bpe = ByteLevelBPETokenizer()
            bpe.train_from_iterator(["synthetic positive input", "synthetic negative input"],
                                    vocab_size=270, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
            bpe.save_model(str(directory))
            tokenizer = RobertaTokenizerFast(vocab_file=str(directory / "vocab.json"),
                                             merges_file=str(directory / "merges.txt"), add_prefix_space=True)
            config = RobertaConfig(vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=1,
                                   num_attention_heads=2, intermediate_size=32,
                                   max_position_embeddings=132, num_labels=2,
                                   hidden_dropout_prob=0, attention_probs_dropout_prob=0)
            model = RobertaForSequenceClassification(config)
            texts = ["synthetic positive input", "synthetic negative input"]
            batch = tokenizer(texts, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
            batch["labels"] = torch.tensor([1, 0])
            before = model.classifier.out_proj.weight.detach().clone()
            train_stage(model, [batch], 1, 1e-3, torch.device("cpu"))
            self.assertFalse(torch.equal(before, model.classifier.out_proj.weight))
            model.eval()
            with torch.inference_mode():
                expected = model(**{k: v for k, v in batch.items() if k != "labels"}).logits
            model.save_pretrained(directory)
            tokenizer.save_pretrained(directory)
            actual, _ = evaluation.predict_logits(directory, texts, 2, torch.device("cpu"))
            torch.testing.assert_close(actual, expected)

            # Exercise the real Trainer/BCE/save path using a tiny local backbone.
            backbone = directory / "backbone"
            RobertaModel(config, add_pooling_layer=False).save_pretrained(backbone)
            tokenizer.save_pretrained(backbone)
            csv = directory / "genre.csv"
            order = list(reversed(GENRES["Book"]))
            pd.DataFrame({"review": texts, **{g: [0, 1] for g in order}}).to_csv(csv, index=False)
            with patch.dict(os.environ, {"RO3_BOOK_GENRE_CSV": str(csv),
                                          "RO3_ROUTER_MODELS": str(directory / "routers")}), \
                    patch.object(router_training, "MODEL_NAME", str(backbone)), \
                    patch.object(torch.cuda, "is_available", return_value=False):
                router_training.train_router("Book", mode="full")
                routed, routed_config = evaluation.predict_logits(
                    router_path("Book"), texts, 2, torch.device("cpu"), router=True)
                self.assertEqual(resolve_genre_order(routed_config, GENRES["Book"]), order)
                self.assertEqual(tuple(routed.shape), (2, 10))


if __name__ == "__main__":
    unittest.main()
