import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


class ClassifierConfigEnvironmentTests(unittest.TestCase):
    def test_dotenv_dimensions_are_loaded_before_config_is_captured(self) -> None:
        module_name = "tests._isolated_classifier_config"
        module_path = Path(__file__).parents[1] / "app" / "classifier_config.py"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)

        with (
            patch.dict(os.environ, {}, clear=False),
            patch.dict(sys.modules, {module_name: module}),
        ):
            os.environ.pop("HF_EMBEDDING_DIMS", None)

            def load_test_dotenv() -> None:
                os.environ["HF_EMBEDDING_DIMS"] = "3072"

            with patch("dotenv.load_dotenv", side_effect=load_test_dotenv) as loader:
                spec.loader.exec_module(module)

        loader.assert_called_once_with()
        self.assertEqual(module.DEFAULT_EMBEDDING_DIMS, 3072)
        self.assertEqual(
            {config["embed_dims"] for config in module.CLASSIFIER_CONFIG.values()},
            {3072},
        )
