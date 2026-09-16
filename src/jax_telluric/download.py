"""Download the external AER spectroscopy data used by jax-telluric."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
from urllib.request import Request, urlopen


MT_CKD_URL = (
    "https://raw.githubusercontent.com/AER-RC/MT_CKD_H2O/4.3/"
    "data/absco-ref_wv-mt-ckd.nc"
)
MT_CKD_SHA256 = "69944eb8b045c268e2daeb2cddf536b99f9e9efe7e91c425067b46a5b287ddb3"
AER_LINES_URL = "https://zenodo.org/records/18881607/files/aer_v_3.9.tgz?download=1"
AER_LINES_SHA256 = "3a55d5deb9430894ce61e1a23475ad1ed7ef01f0a5c527ee6f08d9e5676a0e4e"
AER_LICENSE_URL = (
    "https://raw.githubusercontent.com/AER-RC/AER_Line_File/"
    "4de5983ac4d909f674d2ee502ac667c27f79942f/LICENSE.md"
)
AER_LICENSE_SHA256 = "eed60bbf029b81fd0abd15ef48f0fb65176906a784a34f9eb7652bbc11646ddd"


def default_data_directory() -> Path:
    """Return the configured per-user data directory."""

    configured = os.environ.get("JAX_TELLURIC_DATA")
    if configured:
        return Path(configured).expanduser()
    xdg_data = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg_data).expanduser() if xdg_data else Path.home() / ".local/share"
    return base / "jax-telluric"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, destination: Path, expected_sha256: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and _sha256(destination) == expected_sha256:
        print(f"Using verified {destination}")
        return destination

    partial = destination.with_name(destination.name + ".part")
    if partial.is_file() and _sha256(partial) == expected_sha256:
        partial.replace(destination)
        print(f"Using verified {destination}")
        return destination
    offset = partial.stat().st_size if partial.is_file() else 0
    headers = {"User-Agent": "jax-telluric-data-downloader/0.1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    print(f"Downloading {url}")
    with urlopen(Request(url, headers=headers), timeout=60) as response:
        append = offset > 0 and getattr(response, "status", None) == 206
        if not append:
            offset = 0
        content_length = response.headers.get("Content-Length")
        total = offset + int(content_length) if content_length is not None else None
        mode = "ab" if append else "wb"
        downloaded = offset
        next_report = downloaded + 32 * 1024 * 1024
        with partial.open(mode) as output:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                output.write(block)
                downloaded += len(block)
                if downloaded >= next_report:
                    if total:
                        print(f"  {downloaded / 1024**2:.0f}/{total / 1024**2:.0f} MiB")
                    else:
                        print(f"  {downloaded / 1024**2:.0f} MiB")
                    next_report += 32 * 1024 * 1024
    if total is not None and downloaded != total:
        raise RuntimeError(
            f"incomplete download from {url}: expected {total} bytes, got {downloaded}"
        )
    actual_sha256 = _sha256(partial)
    if actual_sha256 != expected_sha256:
        partial.unlink()
        raise RuntimeError(
            f"SHA-256 mismatch for {url}: expected {expected_sha256}, got {actual_sha256}"
        )
    partial.replace(destination)
    print(f"Verified {destination}")
    return destination


def download_mt_ckd(directory: str | Path | None = None) -> Path:
    """Download the official MT_CKD 4.3 H2O coefficient file."""

    root = default_data_directory() if directory is None else Path(directory).expanduser()
    destination = root / "mt_ckd" / "absco-ref_wv-mt-ckd.nc"
    return _download(MT_CKD_URL, destination, MT_CKD_SHA256)


def _extract_aer_archive(archive: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="aer-lines-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        with tarfile.open(archive, "r:gz") as bundle:
            for member in bundle.getmembers():
                relative = PurePosixPath(member.name)
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or not relative.parts
                    or relative.parts[0] != "aer_v_3.9"
                    or member.issym()
                    or member.islnk()
                    or not (member.isdir() or member.isfile())
                ):
                    raise RuntimeError(f"unsafe path or file type in AER archive: {member.name}")
                target = staging.joinpath(*relative.parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                source = bundle.extractfile(member)
                if source is None:
                    raise RuntimeError(f"could not read AER archive member: {member.name}")
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
        extracted = staging / "aer_v_3.9"
        required = extracted / "line_files_By_Molecule/01_H2O/01_H2O"
        if not required.is_file():
            raise RuntimeError("AER archive does not contain the expected H2O line file")
        extracted.replace(destination)


def download_aer_lines(directory: str | Path | None = None) -> Path:
    """Download and safely extract the official AER line file 3.9 archive."""

    root = default_data_directory() if directory is None else Path(directory).expanduser()
    destination = root / "aer_v_3.9"
    required = destination / "line_files_By_Molecule/01_H2O/01_H2O"
    if required.is_file():
        _download(AER_LICENSE_URL, destination / "LICENSE.md", AER_LICENSE_SHA256)
        print(f"Using existing {destination}")
        return destination
    if destination.exists():
        raise RuntimeError(
            f"incomplete AER line directory exists at {destination}; move or remove it and retry"
        )
    archive = _download(
        AER_LINES_URL, root / "downloads/aer_v_3.9.tgz", AER_LINES_SHA256
    )
    print(f"Extracting {archive}")
    _extract_aer_archive(archive, destination)
    _download(AER_LICENSE_URL, destination / "LICENSE.md", AER_LICENSE_SHA256)
    print(f"Installed AER line files at {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download external spectroscopy data used by jax-telluric"
    )
    parser.add_argument("dataset", choices=("mt-ckd", "aer-lines", "all"))
    parser.add_argument(
        "--output",
        type=Path,
        default=default_data_directory(),
        help="data root (default: JAX_TELLURIC_DATA or the user data directory)",
    )
    args = parser.parse_args()
    if args.dataset in ("mt-ckd", "all"):
        print(f"MT_CKD: {download_mt_ckd(args.output)}")
    if args.dataset in ("aer-lines", "all"):
        print(f"AER lines: {download_aer_lines(args.output)}")


if __name__ == "__main__":
    main()
