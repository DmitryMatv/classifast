import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

from app.qdrant_schema import (
    QdrantValidationIssue,
    QdrantValidationReport,
)
from utilities import sync_payload_indexes


class QdrantCliTests(unittest.TestCase):
    def test_no_argument_invocation_exits_two_without_connecting(self) -> None:
        with (
            patch.object(sync_payload_indexes, "create_qdrant_client") as create_client,
            patch.object(sync_payload_indexes, "load_dotenv") as load_dotenv,
            self.assertRaises(SystemExit) as ctx,
        ):
            sync_payload_indexes.main([])

        self.assertEqual(ctx.exception.code, 2)
        create_client.assert_not_called()
        load_dotenv.assert_not_called()

    def test_check_is_read_only_and_closes_client(self) -> None:
        client = MagicMock()
        report = QdrantValidationReport({"products": False}, ())
        with (
            patch.object(
                sync_payload_indexes, "create_qdrant_client", return_value=client
            ),
            patch.object(
                sync_payload_indexes,
                "inspect_configured_collections",
                return_value=report,
            ) as inspect_collections,
        ):
            result = sync_payload_indexes.main(["check"])

        self.assertEqual(result, 0)
        inspect_collections.assert_called_once_with(client, collection_names=None)
        client.create_payload_index.assert_not_called()
        client.delete_payload_index.assert_not_called()
        client.batch_update_points.assert_not_called()
        client.close.assert_called_once_with()

    def test_check_returns_one_for_contract_violations(self) -> None:
        client = MagicMock()
        report = QdrantValidationReport(
            {},
            (
                QdrantValidationIssue(
                    "products", "missing_payload_index", "missing class_name"
                ),
            ),
        )
        with (
            patch.object(
                sync_payload_indexes, "create_qdrant_client", return_value=client
            ),
            patch.object(
                sync_payload_indexes,
                "inspect_configured_collections",
                return_value=report,
            ),
        ):
            result = sync_payload_indexes.main(["check"])

        self.assertEqual(result, 1)
        client.close.assert_called_once_with()

    def test_collection_filter_is_forwarded_to_check(self) -> None:
        configured = sync_payload_indexes.get_all_collection_names()[0]
        client = MagicMock()
        report = QdrantValidationReport({configured: False}, ())
        with (
            patch.object(
                sync_payload_indexes, "create_qdrant_client", return_value=client
            ),
            patch.object(
                sync_payload_indexes,
                "inspect_configured_collections",
                return_value=report,
            ) as inspect_collections,
        ):
            result = sync_payload_indexes.main(["check", "--collection", configured])

        self.assertEqual(result, 0)
        inspect_collections.assert_called_once_with(
            client, collection_names={configured}
        )

    def test_unknown_collection_exits_two_without_connecting(self) -> None:
        with (
            patch.object(sync_payload_indexes, "create_qdrant_client") as create_client,
            patch.object(sync_payload_indexes, "load_dotenv") as load_dotenv,
            self.assertRaises(SystemExit) as ctx,
        ):
            sync_payload_indexes.main(["check", "--collection", "not-configured"])

        self.assertEqual(ctx.exception.code, 2)
        create_client.assert_not_called()
        load_dotenv.assert_not_called()

    @patch.object(sync_payload_indexes, "create_shared_qdrant_client")
    def test_maintenance_client_uses_120_second_timeout(self, create_shared) -> None:
        sync_payload_indexes.create_qdrant_client()

        create_shared.assert_called_once_with(timeout=120)

    def test_environment_is_loaded_before_client_creation(self) -> None:
        for command in ("check", "apply"):
            with self.subTest(command=command):
                events: list[str] = []
                client = MagicMock()
                report = QdrantValidationReport({}, ())
                with (
                    patch.object(
                        sync_payload_indexes,
                        "load_dotenv",
                        side_effect=lambda: events.append("dotenv"),
                    ),
                    patch.object(
                        sync_payload_indexes,
                        "create_qdrant_client",
                        side_effect=lambda: events.append("client") or client,
                    ),
                    patch.object(
                        sync_payload_indexes,
                        "migrate_configured_collections",
                        return_value=(1, 0),
                    ),
                    patch.object(
                        sync_payload_indexes,
                        "inspect_configured_collections",
                        return_value=report,
                    ),
                ):
                    result = sync_payload_indexes.main([command])

                self.assertEqual(result, 0)
                self.assertEqual(events[:2], ["dotenv", "client"])

    def test_apply_collection_filters_are_deduplicated_and_forwarded(self) -> None:
        configured = sync_payload_indexes.get_all_collection_names()[0]
        selected = {configured}
        client = MagicMock()
        report = QdrantValidationReport({configured: False}, ())
        with (
            patch.object(sync_payload_indexes, "load_dotenv"),
            patch.object(
                sync_payload_indexes, "create_qdrant_client", return_value=client
            ),
            patch.object(
                sync_payload_indexes,
                "migrate_configured_collections",
                return_value=(1, 0),
            ) as migrate,
            patch.object(
                sync_payload_indexes,
                "inspect_configured_collections",
                return_value=report,
            ) as inspect_collections,
        ):
            result = sync_payload_indexes.main(
                [
                    "apply",
                    "--collection",
                    configured,
                    "--collection",
                    configured,
                ]
            )

        self.assertEqual(result, 0)
        migrate.assert_called_once_with(client, collection_names=selected)
        inspect_collections.assert_called_once_with(client, collection_names=selected)

    def test_final_validation_failure_makes_apply_nonzero(self) -> None:
        client = MagicMock()
        invalid_report = QdrantValidationReport(
            {},
            (QdrantValidationIssue("products", "missing_collection", "missing"),),
        )
        with (
            patch.object(sync_payload_indexes, "load_dotenv"),
            patch.object(
                sync_payload_indexes, "create_qdrant_client", return_value=client
            ),
            patch.object(
                sync_payload_indexes,
                "migrate_configured_collections",
                return_value=(1, 0),
            ),
            patch.object(
                sync_payload_indexes,
                "inspect_configured_collections",
                return_value=invalid_report,
            ),
        ):
            result = sync_payload_indexes.main(["apply"])

        self.assertEqual(result, 1)
        client.close.assert_called_once_with()

    def test_operation_exceptions_return_one_and_close_client(self) -> None:
        for command, operation_name in (("check", "run_check"), ("apply", "run_apply")):
            with self.subTest(command=command):
                client = MagicMock()
                with (
                    patch.object(sync_payload_indexes, "load_dotenv"),
                    patch.object(
                        sync_payload_indexes,
                        "create_qdrant_client",
                        return_value=client,
                    ),
                    patch.object(
                        sync_payload_indexes,
                        operation_name,
                        side_effect=RuntimeError("operation failed"),
                    ),
                ):
                    with redirect_stdout(io.StringIO()) as output:
                        result = sync_payload_indexes.main([command])

                self.assertEqual(result, 1)
                self.assertIn("Qdrant operation failed", output.getvalue())
                client.close.assert_called_once_with()

    def test_client_creation_failure_returns_one_without_cleanup(self) -> None:
        with (
            patch.object(sync_payload_indexes, "load_dotenv"),
            patch.object(
                sync_payload_indexes,
                "create_qdrant_client",
                side_effect=RuntimeError("connect failed"),
            ),
        ):
            with redirect_stdout(io.StringIO()) as output:
                result = sync_payload_indexes.main(["check"])

        self.assertEqual(result, 1)
        self.assertIn("connect failed", output.getvalue())

    def test_close_failure_forces_nonzero_without_raising(self) -> None:
        client = MagicMock()
        client.close.side_effect = RuntimeError("close failed")
        with (
            patch.object(sync_payload_indexes, "load_dotenv"),
            patch.object(
                sync_payload_indexes, "create_qdrant_client", return_value=client
            ),
            patch.object(sync_payload_indexes, "run_check", return_value=0),
        ):
            with redirect_stdout(io.StringIO()) as output:
                result = sync_payload_indexes.main(["check"])

        self.assertEqual(result, 1)
        self.assertIn("Qdrant client cleanup failed", output.getvalue())

    def test_operation_and_close_failures_are_both_reported(self) -> None:
        client = MagicMock()
        client.close.side_effect = RuntimeError("close failed")
        with (
            patch.object(sync_payload_indexes, "load_dotenv"),
            patch.object(
                sync_payload_indexes, "create_qdrant_client", return_value=client
            ),
            patch.object(
                sync_payload_indexes,
                "run_apply",
                side_effect=RuntimeError("apply failed"),
            ),
        ):
            with redirect_stdout(io.StringIO()) as output:
                result = sync_payload_indexes.main(["apply"])

        self.assertEqual(result, 1)
        self.assertIn("apply failed", output.getvalue())
        self.assertIn("close failed", output.getvalue())
