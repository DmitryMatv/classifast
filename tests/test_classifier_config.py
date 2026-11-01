import importlib
import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


class ClassifierConfigEnvironmentTests(unittest.TestCase):
    def test_package_loads_dotenv_before_any_module_reads_settings(self) -> None:
        app_dir = Path(__file__).parents[1] / "app"
        package_name = "tests._isolated_app"
        package_spec = importlib.util.spec_from_file_location(
            package_name,
            app_dir / "__init__.py",
            submodule_search_locations=[str(app_dir)],
        )
        self.assertIsNotNone(package_spec)
        self.assertIsNotNone(package_spec.loader)
        package = importlib.util.module_from_spec(package_spec)

        with (
            patch.dict(os.environ, {}, clear=False),
            patch.dict(sys.modules, {package_name: package}),
        ):
            os.environ.pop("HF_EMBEDDING_DIMS", None)
            os.environ.pop("REDIS_HOST", None)

            def load_test_dotenv() -> None:
                os.environ["HF_EMBEDDING_DIMS"] = "3072"
                os.environ["REDIS_HOST"] = "redis.from-dotenv"

            with patch("dotenv.load_dotenv", side_effect=load_test_dotenv) as loader:
                package_spec.loader.exec_module(package)
                usage_tracker = importlib.import_module(f"{package_name}.usage_tracker")
                module = importlib.import_module(f"{package_name}.classifier_config")

        loader.assert_called_once_with()
        self.assertEqual(usage_tracker.REDIS_HOST, "redis.from-dotenv")
        self.assertEqual(module.DEFAULT_EMBEDDING_DIMS, 3072)
        self.assertEqual(
            {config["embed_dims"] for config in module.CLASSIFIER_CONFIG.values()},
            {3072},
        )
