"""Mirror-only resolution and quiet, resumable aria2 transfers.

Signed URLs stay in memory and aria2 stdin, never argv or a session/log file.
Only the target and its binary .aria2 progress file are written by aria2.
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

MIRROR_HOST = "hf-mirror.com"
_REDIRECT_HOSTS = {MIRROR_HOST, "cas-bridge.xethub.hf.co", "cdn-lfs.hf.co",
                   "cdn-lfs-us-1.hf.co", "cdn-lfs-eu-1.hf.co", "cdn-lfs.hf-mirror.com"}


def _validate_url(url: str, *, initial: bool = False) -> None:
    parsed = urllib.parse.urlsplit(url)
    hosts = {MIRROR_HOST} if initial else _REDIRECT_HOSTS
    if (any(ord(char) <= 32 for char in url) or parsed.scheme != "https"
            or parsed.hostname not in hosts or parsed.username or parsed.password
            or parsed.fragment or parsed.port not in (None, 443)
            or (initial and "/resolve/" not in parsed.path)):
        raise ValueError("Only mirror resolve URLs and approved HTTPS storage redirects are allowed")


def _sha_header(value: str | None) -> str | None:
    if value:
        value = value.strip('"')
        if re.fullmatch(r"[a-fA-F0-9]{64}", value):
            return value.lower()
    return None


class _MirrorRedirects(urllib.request.HTTPRedirectHandler):
    def __init__(self):
        super().__init__()
        self.linked_size = None
        self.linked_sha256 = None

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validate_url(newurl)
        if urllib.parse.urlsplit(req.full_url).hostname == MIRROR_HOST:
            size = headers.get("X-Linked-Size")
            if size and size.isdigit():
                self.linked_size = int(size)
            self.linked_sha256 = _sha_header(headers.get("X-Linked-Etag")) or self.linked_sha256
        result = super().redirect_request(req, fp, code, msg, headers, newurl)
        # Python 3.10's redirect handler turns HEAD into GET. Metadata lookup
        # must never start streaming multi-gigabyte model weights.
        if result and req.get_method() == "HEAD":
            result.method = "HEAD"
        if result and urllib.parse.urlsplit(newurl).hostname != MIRROR_HOST:
            result.remove_header("Authorization")
        return result


@dataclass(frozen=True)
class ResolvedDownload:
    url: str = field(repr=False)
    size: int
    sha256: str | None = None


def resolve_download(url: str, no_proxy: bool = False) -> ResolvedDownload:
    """HEAD the mirror and approved redirects; reject unknown content length."""
    _validate_url(url, initial=True)
    redirects = _MirrorRedirects()
    handlers = [redirects]
    if no_proxy or os.environ.get("POT_DISABLE_PROXY") == "1":
        handlers.append(urllib.request.ProxyHandler({}))
    opener = urllib.request.build_opener(*handlers)
    headers = {"User-Agent": "pot-reproduction/1.0", "Accept-Encoding": "identity"}
    if os.environ.get("HF_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["HF_TOKEN"]
    try:
        with opener.open(urllib.request.Request(url, headers=headers, method="HEAD"), timeout=30) as response:
            final_url = response.geturl()
            _validate_url(final_url)
            length = response.headers.get("Content-Length")
            final_size = int(length) if length and length.isdigit() else None
            expected = redirects.linked_size if redirects.linked_size is not None else final_size
            if expected is None or expected < 0:
                raise ValueError("Missing file length")
            if final_size is not None and final_size != expected:
                raise ValueError("Conflicting file lengths")
            digest = redirects.linked_sha256 or _sha_header(response.headers.get("X-Linked-Etag")) or _sha_header(response.headers.get("ETag"))
            return ResolvedDownload(final_url, expected, digest)
    except Exception:
        # HTTPError and TLS errors can contain the signed URL or request details.
        raise RuntimeError("Mirror metadata resolution failed; no alternate hub was contacted") from None


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aria2_download(url: str, destination: str | Path, no_proxy: bool = False) -> int:
    """Transfer one mirror file with 8 streams; retain partial data on failure.

    Returns validated byte count. If metadata supplies a SHA-256, validate it as
    well. Callers should additionally check their pinned expected checksum.
    """
    no_proxy = no_proxy or os.environ.get("POT_DISABLE_PROXY") == "1"
    raw_destination = Path(destination)
    if raw_destination.is_symlink() or raw_destination.with_name(raw_destination.name + ".aria2").is_symlink():
        raise ValueError("Download target and progress file must not be symlinks")
    destination = raw_destination.resolve()
    if any(ord(char) < 32 for char in str(destination)) or destination.name in ("", ".", ".."):
        raise ValueError("Invalid download destination")
    destination.parent.mkdir(parents=True, exist_ok=True)
    resolved = resolve_download(url, no_proxy=no_proxy)
    environment = dict(os.environ)
    for key in list(environment):
        if key.upper() in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"} or (no_proxy and key.lower().endswith("_proxy")):
            environment.pop(key)
    command = ["aria2c", "--no-conf=true", "--no-netrc=true", "--input-file=-",
               "--continue=true", "--always-resume=true", "--auto-file-renaming=false",
               "--allow-overwrite=false", "--file-allocation=none", "-x8", "-s8", "-j1", "-k1M",
               "--max-tries=5", "--retry-wait=3", "--connect-timeout=20", "--timeout=60",
               "--check-certificate=true", "--http-accept-gzip=false", "--content-disposition=false",
               "--follow-torrent=false", "--follow-metalink=false", "--enable-rpc=false",
               "--summary-interval=0", "--show-console-readout=false", "--download-result=hide",
               "--quiet=true", "--log=", "--auto-save-interval=5"]
    # --out on argv is ignored when --input-file is used: specify dir/out as
    # per-entry options. Newlines/control characters in URL/path were rejected.
    entry = f"{resolved.url}\n  dir={destination.parent}\n  out={destination.name}\n"
    try:
        result = subprocess.run(command, input=entry, text=True, encoding="utf-8", env=environment,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except OSError:
        raise RuntimeError("aria2c is unavailable; install it or omit --aria2") from None
    if result.returncode != 0:
        raise RuntimeError(f"aria2 transfer failed (exit {result.returncode}); partial data retained for resume")
    if not destination.is_file() or destination.stat().st_size != resolved.size:
        raise RuntimeError("aria2 transfer length mismatch; partial data retained")
    if resolved.sha256 and _digest(destination) != resolved.sha256:
        raise RuntimeError("aria2 transfer SHA-256 mismatch; downloaded data retained for inspection")
    return resolved.size
