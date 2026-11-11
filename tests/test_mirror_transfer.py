"""No network or aria2 installation required: verify the transfer boundary."""
import hashlib
import os
import subprocess
import unittest
import urllib.request
from types import SimpleNamespace
from unittest.mock import patch

from scripts.mirror_transfer import (_MirrorRedirects, _validate_url, ResolvedDownload,
                                     aria2_download, resolve_download)
from tests.test_data_eval import scratch


MIRROR_URL = "https://hf-mirror.com/datasets/livecodebench/code_generation_lite/resolve/main/test6.jsonl"
SIGNED_URL = "https://cas-bridge.xethub.hf.co/object?Signature=PRIVATE_SIGNATURE"


class MirrorTransferTests(unittest.TestCase):
    def test_redirect_removes_auth_and_preserves_head(self):
        handler = _MirrorRedirects()
        request = urllib.request.Request(MIRROR_URL, headers={"Authorization": "Bearer SECRET"}, method="HEAD")
        result = handler.redirect_request(request, None, 302, "Found",
                                          {"X-Linked-Size": "42", "X-Linked-Etag": '"' + "a" * 64 + '"'}, SIGNED_URL)
        self.assertEqual(result.get_method(), "HEAD")
        self.assertIsNone(result.get_header("Authorization"))
        self.assertEqual(handler.linked_size, 42)
        self.assertEqual(handler.linked_sha256, "a" * 64)

    def test_arbitrary_origins_and_line_injection_rejected(self):
        for url in ("https://huggingface.co/repo/resolve/main/file", "https://example.org/a",
                    SIGNED_URL, MIRROR_URL + "\n  out=bad", "http://hf-mirror.com/r/resolve/main/a"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                _validate_url(url, initial=True)
        with self.assertRaises(ValueError):
            _MirrorRedirects().redirect_request(urllib.request.Request(MIRROR_URL), None, 302, "Found", {}, "https://example.org/secret")

    def test_python310_redirect_get_is_forced_back_to_head(self):
        request = urllib.request.Request(MIRROR_URL, method="HEAD")
        legacy_result = urllib.request.Request(SIGNED_URL, method="GET")
        with patch("urllib.request.HTTPRedirectHandler.redirect_request", return_value=legacy_result):
            result = _MirrorRedirects().redirect_request(request, None, 302, "Found", {}, SIGNED_URL)
        self.assertEqual(result.get_method(), "HEAD")

    def test_metadata_resolution_errors_do_not_echo_signed_urls(self):
        with patch("scripts.mirror_transfer.urllib.request.build_opener") as opener:
            opener.return_value.open.side_effect = OSError(SIGNED_URL)
            with self.assertRaises(RuntimeError) as caught:
                resolve_download(MIRROR_URL)
        self.assertNotIn("PRIVATE_SIGNATURE", str(caught.exception))
        self.assertNotIn("PRIVATE_SIGNATURE", repr(ResolvedDownload(SIGNED_URL, 42)))

    def test_stdin_only_uri_no_child_token_no_proxy_and_exact_target(self):
        content = b"verified bytes"
        digest = hashlib.sha256(content).hexdigest()
        with scratch() as directory:
            target = directory / "file.part"
            def run(command, **kwargs):
                self.assertNotIn(SIGNED_URL, " ".join(command))
                self.assertNotIn("PRIVATE_SIGNATURE", " ".join(command))
                self.assertIn("--input-file=-", command)
                self.assertIn("--no-conf=true", command)
                self.assertIn("--no-netrc=true", command)
                self.assertIn("-x8", command)
                self.assertIn("-s8", command)
                self.assertEqual(kwargs["input"], f"{SIGNED_URL}\n  dir={target.parent}\n  out=file.part\n")
                self.assertNotIn("HF_TOKEN", kwargs["env"])
                self.assertNotIn("HTTPS_PROXY", kwargs["env"])
                self.assertNotIn("all_proxy", kwargs["env"])
                self.assertEqual(kwargs["stdout"], subprocess.DEVNULL)
                self.assertEqual(kwargs["stderr"], subprocess.DEVNULL)
                target.write_bytes(content)
                return SimpleNamespace(returncode=0)
            with patch.dict(os.environ, {"HF_TOKEN": "SECRET", "HTTPS_PROXY": "http://proxy", "all_proxy": "http://proxy"}), \
                 patch("scripts.mirror_transfer.resolve_download", return_value=ResolvedDownload(SIGNED_URL, len(content), digest)), \
                 patch("scripts.mirror_transfer.subprocess.run", side_effect=run):
                self.assertEqual(aria2_download(MIRROR_URL, target, no_proxy=True), len(content))

    def test_failure_keeps_partial_and_redacts_details(self):
        with scratch() as directory:
            target = directory / "file.part"
            target.write_bytes(b"partial")
            with patch("scripts.mirror_transfer.resolve_download", return_value=ResolvedDownload(SIGNED_URL, 100)), \
                 patch("scripts.mirror_transfer.subprocess.run", return_value=SimpleNamespace(returncode=7)):
                with self.assertRaisesRegex(RuntimeError, "partial data retained"):
                    aria2_download(MIRROR_URL, target)
            self.assertEqual(target.read_bytes(), b"partial")

    def test_size_and_sha_mismatch_rejected(self):
        with scratch() as directory:
            target = directory / "file.part"
            target.write_bytes(b"abc")
            with patch("scripts.mirror_transfer.subprocess.run", return_value=SimpleNamespace(returncode=0)):
                with patch("scripts.mirror_transfer.resolve_download", return_value=ResolvedDownload(SIGNED_URL, 4)):
                    with self.assertRaisesRegex(RuntimeError, "length mismatch"):
                        aria2_download(MIRROR_URL, target)
                with patch("scripts.mirror_transfer.resolve_download", return_value=ResolvedDownload(SIGNED_URL, 3, "0" * 64)):
                    with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                        aria2_download(MIRROR_URL, target)


if __name__ == "__main__":
    unittest.main()
