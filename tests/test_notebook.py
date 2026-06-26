import ast
import json
import unittest

from src.utils import project_path


class NotebookTests(unittest.TestCase):
    def test_colab_runner_code_cells_parse_and_outputs_are_clear(self):
        path = project_path("notebooks", "01_colab_runner.ipynb")
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            self.assertEqual(cell.get("outputs", []), [], f"cell {index} has saved outputs")
            self.assertIsNone(cell.get("execution_count"), f"cell {index} has execution_count")
            source = "".join(cell["source"])
            ast.parse(source, filename=f"{path}:cell-{index}")


if __name__ == "__main__":
    unittest.main()
