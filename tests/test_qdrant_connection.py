import os
import unittest
from unittest.mock import patch

from app import qdrant_connection


class QdrantConnectionTests(unittest.TestCase):
    @patch.dict(
        os.environ,
        {
            "QDRANT_API_KEY": "",
            "QDRANT_HOST": "qdrant-eu2dfebw2y9vs524c2lrlnhw",
        },
        clear=True,
    )
    @patch.object(qdrant_connection, "QdrantClient")
    def test_on_prem_host_uses_local_http_and_default_port(self, client_class) -> None:
        qdrant_connection.create_qdrant_client(timeout=30)

        client_class.assert_called_once_with(
            url="http://qdrant-eu2dfebw2y9vs524c2lrlnhw:6333",
            api_key=None,
            timeout=30,
        )

    @patch.dict(
        os.environ,
        {
            "QDRANT_API_KEY": "",
            "QDRANT_URL": "qdrant-eu2.classifast.com",
        },
        clear=True,
    )
    @patch.object(qdrant_connection, "QdrantClient")
    def test_remote_bare_url_uses_https_default_port(self, client_class) -> None:
        qdrant_connection.create_qdrant_client(timeout=120)

        client_class.assert_called_once_with(
            url="https://qdrant-eu2.classifast.com",
            port=443,
            api_key=None,
            timeout=120,
        )

    @patch.dict(os.environ, {"QDRANT_URL": "http://example.test"}, clear=True)
    @patch.object(qdrant_connection, "QdrantClient")
    def test_portless_full_http_url_uses_standard_port(self, client_class) -> None:
        qdrant_connection.create_qdrant_client(timeout=30)

        client_class.assert_called_once_with(
            url="http://example.test",
            port=80,
            api_key=None,
            timeout=30,
        )

    @patch.dict(os.environ, {"QDRANT_URL": "https://example.test:7443/"}, clear=True)
    def test_preserves_full_https_url(self) -> None:
        self.assertEqual(
            qdrant_connection.resolve_qdrant_url(), "https://example.test:7443"
        )

    @patch.dict(os.environ, {"QDRANT_URL": "http://qdrant:6333"}, clear=True)
    def test_preserves_full_http_url(self) -> None:
        self.assertEqual(qdrant_connection.resolve_qdrant_url(), "http://qdrant:6333")

    @patch.dict(os.environ, {"QDRANT_URL": "http://qdrant:6333"}, clear=True)
    @patch.object(qdrant_connection, "QdrantClient")
    def test_explicit_http_port_is_not_overridden(self, client_class) -> None:
        qdrant_connection.create_qdrant_client(timeout=30)

        client_class.assert_called_once_with(
            url="http://qdrant:6333",
            api_key=None,
            timeout=30,
        )

    @patch.dict(os.environ, {"QDRANT_URL": "remote.example.test"}, clear=True)
    def test_bare_remote_hostname_defaults_to_https(self) -> None:
        self.assertEqual(
            qdrant_connection.resolve_qdrant_url(), "https://remote.example.test"
        )

    @patch.dict(
        os.environ,
        {
            "QDRANT_URL": "remote.example.test",
            "QDRANT_HOST": "ignored",
            "QDRANT_PORT": "9999",
        },
        clear=True,
    )
    def test_url_takes_precedence_over_host_and_port(self) -> None:
        self.assertEqual(
            qdrant_connection.resolve_qdrant_url(), "https://remote.example.test"
        )

    @patch.dict(
        os.environ,
        {"QDRANT_HOST": "qdrant", "QDRANT_PORT": "6334"},
        clear=True,
    )
    def test_falls_back_to_host_and_port(self) -> None:
        self.assertEqual(qdrant_connection.resolve_qdrant_url(), "http://qdrant:6334")

    @patch.dict(
        os.environ,
        {"QDRANT_URL": "qdrant.example", "QDRANT_API_KEY": "secret"},
        clear=True,
    )
    @patch.object(qdrant_connection, "QdrantClient")
    def test_client_factory_applies_api_key_and_timeout(self, client_class) -> None:
        qdrant_connection.create_qdrant_client(timeout=120)

        client_class.assert_called_once_with(
            url="https://qdrant.example",
            port=443,
            api_key="secret",
            timeout=120,
        )

    @patch.dict(os.environ, {}, clear=True)
    @patch.object(qdrant_connection, "QdrantClient")
    def test_client_factory_omits_blank_api_key(self, client_class) -> None:
        qdrant_connection.create_qdrant_client(timeout=30)

        client_class.assert_called_once_with(
            url="http://localhost:6333",
            api_key=None,
            timeout=30,
        )
