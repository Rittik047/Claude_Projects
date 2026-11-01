"""Resumable, checksum-verified downloads from the PhysioNet AWS open-data mirror.

The cloud proxy can cut a transfer mid-file, so every file is written to `<dest>.part`,
resumed with an HTTP Range request, and only renamed into place after its SHA-256 matches
the dataset's published `SHA256SUMS.txt`.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

from ecgtrust.config import Config

DatasetName = Literal["ptbxl", "ptbxl_plus", "external"]
CHUNK = 1 << 20
_TRANSIENT = (urllib.error.URLError, http.client.HTTPException, ConnectionError, TimeoutError, OSError)


class ChecksumError(Exception):
    """Downloaded bytes do not match the published SHA-256."""


class _Truncated(Exception):
    pass


def parse_sha256sums(text: str) -> dict[str, str]:
    """Parse `<hex digest> <relative path>` lines into {relative path: digest}."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            out[parts[1].lstrip("*")] = parts[0]
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _fetch_into(url: str, part: Path, timeout: float) -> None:
    start = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-"} if start else {})
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        if e.code == 416 and start:  # nothing left to fetch; let the checksum decide
            return
        raise
    with resp:
        if start and resp.status != 206:  # server ignored Range: restart from scratch
            start = 0
        expected = resp.headers.get("Content-Length")
        received = 0
        with part.open("ab" if start else "wb") as f:
            while True:
                try:
                    block = resp.read(CHUNK)
                except http.client.IncompleteRead as e:
                    f.write(e.partial)
                    raise _Truncated from e
                if not block:
                    break
                f.write(block)
                received += len(block)
        if expected is not None and received < int(expected):
            raise _Truncated


def download_file(
    url: str,
    dest: Path,
    sha256: str | None,
    retries: int = 6,
    backoff: float = 1.0,
    timeout: float = 60.0,
) -> Path:
    """Download `url` to `dest`, resuming on cut-offs; verify `sha256` when given."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if sha256 is None or sha256_file(dest) == sha256:
            return dest
        dest.unlink()  # corrupt or stale cache: never trust it
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries + 1):
        try:
            _fetch_into(url, part, timeout)
            break
        except (_Truncated, *_TRANSIENT) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code in (403, 404):
                raise
            if attempt == retries:
                raise
            time.sleep(backoff * 2**attempt)
    if sha256 is not None and sha256_file(part) != sha256:
        part.unlink(missing_ok=True)
        raise ChecksumError(f"SHA-256 mismatch for {url}")
    part.replace(dest)
    return dest


# --- dataset selection (spec §2; plan Task 2) -------------------------------------------

def _ptbxl(p: str) -> bool:
    return p in {"ptbxl_database.csv", "scp_statements.csv"} or p.startswith("records100/")


def _ptbxl_plus(p: str) -> bool:
    return p in {
        "features/12sl_features.csv",
        "features/ecgdeli_features.csv",
        "features/feature_description.csv",
    } or (p.startswith("labels/") and p.count("/") == 1)


def _external(p: str) -> bool:
    return p == "ConditionNames_SNOMED-CT.csv" or p.startswith("WFDBRecords/")


DATASETS: dict[str, tuple[str, Callable[[str], bool]]] = {
    "ptbxl": ("ptb-xl", _ptbxl),
    "ptbxl_plus": ("ptb-xl-plus", _ptbxl_plus),
    "external": ("ecg-arrhythmia", _external),
}


def dataset_base_url(cfg: Config, name: DatasetName) -> str:
    slug, _ = DATASETS[name]
    return f"{cfg.mirror.rstrip('/')}/{slug}/{cfg.versions[name]}/"


def download_dataset(cfg: Config, name: DatasetName, workers: int = 16) -> list[Path]:
    """Download the spec-selected files of one dataset; write `download_log.json`."""
    _, keep = DATASETS[name]
    base = dataset_base_url(cfg, name)
    root = cfg.dataset_dir(name)
    sums_path = download_file(base + "SHA256SUMS.txt", root / "SHA256SUMS.txt", sha256=None)
    sums = {p: h for p, h in parse_sha256sums(sums_path.read_text()).items() if keep(p)}
    t0, done, failures = time.time(), [], []
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(download_file, base + p, root / p, h): p for p, h in sums.items()}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                done.append(fut.result())
            except Exception as e:  # noqa: BLE001 - every failure is logged, then raised
                failures.append({"path": futs[fut], "error": repr(e)})
            if i % 5000 == 0:
                print(f"[{name}] {i}/{len(futs)} files", flush=True)
    log = {
        "dataset": name,
        "version": cfg.versions[name],
        "base_url": base,
        "n_selected": len(sums),
        "n_ok": len(done),
        "n_failed": len(failures),
        "n_bytes": sum(p.stat().st_size for p in done),
        "seconds": round(time.time() - t0, 1),
        "failures": failures[:100],
    }
    (root / "download_log.json").write_text(json.dumps(log, indent=2))
    if failures:
        raise RuntimeError(f"{len(failures)} files failed for {name}; see download_log.json")
    return done


def register(sub) -> None:
    p = sub.add_parser("download", help="download a dataset from the PhysioNet AWS mirror")
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    p.add_argument("--workers", type=int, default=16)

    def run(args):
        from ecgtrust.config import load_config

        cfg = load_config(args.config)
        files = download_dataset(cfg, args.dataset, args.workers)
        print(f"{args.dataset}: {len(files)} files verified")

    p.set_defaults(func=run)
