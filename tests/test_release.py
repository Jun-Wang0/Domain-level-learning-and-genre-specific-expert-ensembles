"""Check published notebooks without executing training or mounting Drive."""

import ast
from pathlib import Path
import unittest

import nbformat


ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_notebook_syntax_and_clean_outputs(self):
        notebooks = list((ROOT / "notebooks").glob("*.ipynb"))
        self.assertEqual(len(notebooks), 4)
        for path in notebooks:
            with self.subTest(notebook=path.name):
                notebook = nbformat.read(path, as_version=4)
                nbformat.validate(notebook)
                for cell in notebook.cells:
                    self.assertEqual(dict(cell.metadata), {})
                    if cell.cell_type == "code":
                        ast.parse(cell.source)
                        self.assertEqual(cell.outputs, [])
                        self.assertIsNone(cell.execution_count)

    def test_main_has_no_auxiliary_model_branch(self):
        paths = [ROOT / name for name in ["train_sentiment.py", "evaluate_cross_domain.py"]]
        texts = [path.read_text(encoding="utf-8") for path in paths]
        notebook = nbformat.read(ROOT / "notebooks/main_book_to_light.ipynb", as_version=4)
        texts.extend(cell.source for cell in notebook.cells if cell.cell_type == "code")
        for source in texts:
            for obsolete_name in ("metaphor_features", "metaphor_labels", "use_metaphor",
                                  "RobertaMetaphor", "Word2Vec", "wordnet"):
                self.assertNotIn(obsolete_name, source)


if __name__ == "__main__":
    unittest.main()
