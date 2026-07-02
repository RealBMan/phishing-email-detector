import base64
import os
import sys

# Make ``src`` importable without needing an installed package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from parse import extract_body_from_string  # noqa: E402


class TestSinglePartEmails:
    def test_simple_plain_text(self):
        raw = (
            "From: alice@example.com\r\n"
            "To: bob@example.com\r\n"
            "Subject: Hello\r\n"
            "Content-Type: text/plain; charset=utf-8\r\n"
            "\r\n"
            "This is the body."
        )
        assert extract_body_from_string(raw) == "This is the body."

    def test_body_without_explicit_content_type(self):
        raw = (
            "Subject: No content type\r\n"
            "\r\n"
            "Plain body defaults to utf-8."
        )
        assert extract_body_from_string(raw) == "Plain body defaults to utf-8."

    def test_empty_body_returns_empty_string(self):
        raw = "Subject: Empty\r\n\r\n"
        assert extract_body_from_string(raw) == ""

    def test_headers_only_no_blank_line(self):
        # No body at all; get_payload(decode=True) yields no content.
        raw = "Subject: Headers only"
        assert extract_body_from_string(raw) == ""

    def test_non_utf8_charset_is_decoded(self):
        body = "Café costs £5"
        encoded = base64.b64encode(body.encode("latin-1")).decode("ascii")
        raw = (
            "Subject: Latin1\r\n"
            "Content-Type: text/plain; charset=latin-1\r\n"
            "Content-Transfer-Encoding: base64\r\n"
            "\r\n"
            f"{encoded}"
        )
        assert extract_body_from_string(raw) == body

    def test_invalid_bytes_are_replaced_not_raised(self):
        # Declares utf-8 but ships invalid bytes; errors="replace" must apply.
        encoded = base64.b64encode(b"valid \xff\xfe end").decode("ascii")
        raw = (
            "Subject: Broken bytes\r\n"
            "Content-Type: text/plain; charset=utf-8\r\n"
            "Content-Transfer-Encoding: base64\r\n"
            "\r\n"
            f"{encoded}"
        )
        result = extract_body_from_string(raw)
        assert result.startswith("valid ")
        assert "�" in result  # replacement character


class TestMultipartEmails:
    def test_multipart_returns_plain_text_part(self):
        raw = (
            "From: alice@example.com\r\n"
            "Subject: Multipart\r\n"
            'Content-Type: multipart/alternative; boundary="BOUND"\r\n'
            "\r\n"
            "--BOUND\r\n"
            "Content-Type: text/plain; charset=utf-8\r\n"
            "\r\n"
            "Plain version.\r\n"
            "--BOUND\r\n"
            "Content-Type: text/html; charset=utf-8\r\n"
            "\r\n"
            "<p>HTML version.</p>\r\n"
            "--BOUND--\r\n"
        )
        assert extract_body_from_string(raw).strip() == "Plain version."

    def test_multipart_skips_attachment_and_finds_plain(self):
        raw = (
            "Subject: With attachment\r\n"
            'Content-Type: multipart/mixed; boundary="BOUND"\r\n'
            "\r\n"
            "--BOUND\r\n"
            "Content-Type: text/plain; charset=utf-8\r\n"
            'Content-Disposition: attachment; filename="notes.txt"\r\n'
            "\r\n"
            "I am an attachment, ignore me.\r\n"
            "--BOUND\r\n"
            "Content-Type: text/plain; charset=utf-8\r\n"
            "\r\n"
            "Real body.\r\n"
            "--BOUND--\r\n"
        )
        assert extract_body_from_string(raw).strip() == "Real body."

    def test_multipart_with_no_plain_text_returns_empty(self):
        raw = (
            "Subject: HTML only\r\n"
            'Content-Type: multipart/alternative; boundary="BOUND"\r\n'
            "\r\n"
            "--BOUND\r\n"
            "Content-Type: text/html; charset=utf-8\r\n"
            "\r\n"
            "<p>Only HTML here.</p>\r\n"
            "--BOUND--\r\n"
        )
        assert extract_body_from_string(raw) == ""

    def test_multipart_returns_first_plain_part(self):
        raw = (
            "Subject: Two plain parts\r\n"
            'Content-Type: multipart/mixed; boundary="BOUND"\r\n'
            "\r\n"
            "--BOUND\r\n"
            "Content-Type: text/plain; charset=utf-8\r\n"
            "\r\n"
            "First.\r\n"
            "--BOUND\r\n"
            "Content-Type: text/plain; charset=utf-8\r\n"
            "\r\n"
            "Second.\r\n"
            "--BOUND--\r\n"
        )
        assert extract_body_from_string(raw).strip() == "First."
