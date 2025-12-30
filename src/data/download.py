"""Download FloorPlanCAD dataset from Google Drive."""

import lzma
import tarfile
import zipfile
from pathlib import Path

import gdown

# Google Drive file IDs extracted from share links
DATASET_FILES = {
    "train1": {
        "id": "1HcyKt6qWeXog-tRfvEjdO3O3TN91PXGL",
        "filename": "train_set_1.zip",
        "description": "Training set part 1",
    },
    "train2": {
        "id": "1kSS7OB_EEu7VJzb0W8DK9_nu1EvshioV",
        "filename": "train_set_2.zip",
        "description": "Training set part 2",
    },
    "test": {
        "id": "1jxpYgxnLUbXEzMOsjaMPQFSuvmvHimiZ",
        "filename": "test_set.zip",
        "description": "Test set",
    },
}


def download_file(file_id: str, output_path: Path, quiet: bool = False) -> bool:
    """Download a file from Google Drive.

    Args:
        file_id: Google Drive file ID.
        output_path: Path to save the downloaded file.
        quiet: Whether to suppress progress output.

    Returns:
        True if download was successful.
    """
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        gdown.download(url, str(output_path), quiet=quiet)
        return output_path.exists()
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def extract_archive(archive_path: Path, extract_dir: Path) -> None:
    """Extract an archive file (zip, tar.xz, or xz).

    Args:
        archive_path: Path to the archive file.
        extract_dir: Directory to extract to.
    """
    print(f"Extracting {archive_path.name}...")

    # Try to detect file type
    with open(archive_path, "rb") as f:
        header = f.read(6)

    # XZ magic bytes: FD 37 7A 58 5A 00
    if header[:6] == b"\xfd7zXZ\x00":
        # XZ compressed - could be tar.xz or just xz
        try:
            with lzma.open(archive_path, "rb") as xz_file:
                # Check if it's a tar archive
                tar_header = xz_file.read(262)
                xz_file.seek(0)
                if len(tar_header) >= 262 and tar_header[257:262] == b"ustar":
                    # It's a tar.xz file
                    with tarfile.open(fileobj=xz_file, mode="r:") as tar:
                        tar.extractall(extract_dir)
                else:
                    # Just xz compressed data, extract as single file
                    xz_file.seek(0)
                    output_path = extract_dir / archive_path.stem
                    with open(output_path, "wb") as out_file:
                        out_file.write(xz_file.read())
        except lzma.LZMAError as e:
            print(f"  Error extracting XZ: {e}")
            # Try as tar.xz directly
            try:
                with tarfile.open(archive_path, "r:xz") as tar:
                    tar.extractall(extract_dir)
            except Exception as e2:
                print(f"  Also failed as tar.xz: {e2}")
                raise
    # ZIP magic bytes: PK (50 4B)
    elif header[:2] == b"PK":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_dir)
    # Tar magic at offset 257
    else:
        # Try as regular tar
        try:
            with tarfile.open(archive_path, "r:*") as tar:
                tar.extractall(extract_dir)
        except tarfile.TarError as err:
            raise ValueError(f"Unknown archive format: {archive_path}") from err

    print(f"Extracted to {extract_dir}")


def download_floorplancad(
    output_dir: Path | str = "data/floorplancad",
    parts: list[str] | None = None,
    extract: bool = True,
    keep_zip: bool = False,
) -> None:
    """Download FloorPlanCAD dataset.

    Args:
        output_dir: Directory to save the dataset.
        parts: Which parts to download. Options: "train1", "train2", "test", or "all".
                Defaults to all parts.
        extract: Whether to extract zip files after download.
        keep_zip: Whether to keep zip files after extraction.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if parts is None or "all" in parts:
        parts = list(DATASET_FILES.keys())

    for part in parts:
        if part not in DATASET_FILES:
            print(f"Unknown part: {part}. Skipping.")
            continue

        info = DATASET_FILES[part]
        zip_path = output_dir / info["filename"]

        print(f"\nDownloading {info['description']}...")
        print(f"  File: {info['filename']}")

        if zip_path.exists():
            print("  File already exists, skipping download.")
        else:
            success = download_file(info["id"], zip_path)
            if not success:
                print(f"  Failed to download {part}")
                continue

        if extract:
            extract_archive(zip_path, output_dir)

            if not keep_zip:
                zip_path.unlink()
                print(f"  Removed {zip_path.name}")

    print("\nDownload complete!")
    print(f"Dataset saved to: {output_dir.absolute()}")


def main() -> None:
    """CLI entry point for downloading dataset."""
    import argparse

    parser = argparse.ArgumentParser(description="Download FloorPlanCAD dataset")
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("data/floorplancad"),
        help="Output directory (default: data/floorplancad)",
    )
    parser.add_argument(
        "--parts",
        "-p",
        nargs="+",
        choices=["train1", "train2", "test", "all"],
        default=["all"],
        help="Which parts to download (default: all)",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Don't extract zip files",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep zip files after extraction",
    )
    args = parser.parse_args()

    download_floorplancad(
        output_dir=args.output_dir,
        parts=args.parts,
        extract=not args.no_extract,
        keep_zip=args.keep_zip,
    )


if __name__ == "__main__":
    main()
