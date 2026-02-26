from pathlib import Path
from typing import Any, Optional

from data.storages.aws_s3 import (
    build_s3_client,
    build_s3_key,
    download_file,
    parse_s3_uri,
)


def resolve_local_data_root_from_data_source(
    data_source: str,
    data_root: str,
    aws_s3_data_root: Optional[str] = None,
) -> Path:
    # Input validations
    assert isinstance(data_source, str), f"{type(data_source)=}"
    assert data_source in {"local", "AWS_S3"}, f"{data_source=}"
    assert isinstance(data_root, str), f"{type(data_root)=}"
    assert data_root != "", f"{data_root=}"
    assert aws_s3_data_root is None or isinstance(
        aws_s3_data_root, str
    ), f"{type(aws_s3_data_root)=}"
    assert data_source == "local" or (
        aws_s3_data_root is not None and aws_s3_data_root != ""
    ), f"{data_source=}, {aws_s3_data_root=}"

    # Input normalizations
    data_root = str(Path(data_root).expanduser().resolve())
    if data_source == "local":
        aws_s3_data_root = ""
    if data_source == "AWS_S3":
        aws_s3_data_root = aws_s3_data_root.rstrip("/")

    if data_source == "local":
        return Path(data_root)
    _, s3_root_prefix = parse_s3_uri(s3_uri=aws_s3_data_root)
    if s3_root_prefix == "":
        return Path(data_root)
    return Path(data_root) / s3_root_prefix


def fetch_file_from_data_source(
    data_source: str,
    data_root: str,
    relative_filepath: str,
    aws_s3_data_root: Optional[str] = None,
    force: bool = False,
    s3_client: Optional[Any] = None,
) -> Path:
    # Input validations
    assert isinstance(data_source, str), f"{type(data_source)=}"
    assert data_source in {"local", "AWS_S3"}, f"{data_source=}"
    assert isinstance(data_root, str), f"{type(data_root)=}"
    assert data_root != "", f"{data_root=}"
    assert isinstance(relative_filepath, str), f"{type(relative_filepath)=}"
    assert relative_filepath != "", f"{relative_filepath=}"
    assert relative_filepath.strip("/") != "", f"{relative_filepath=}"
    assert aws_s3_data_root is None or isinstance(
        aws_s3_data_root, str
    ), f"{type(aws_s3_data_root)=}"
    assert data_source == "local" or (
        aws_s3_data_root is not None and aws_s3_data_root != ""
    ), f"{data_source=}, {aws_s3_data_root=}"
    assert isinstance(force, bool), f"{type(force)=}"

    # Input normalizations
    data_root = str(Path(data_root).expanduser().resolve())
    relative_filepath = relative_filepath.lstrip("/")
    if data_source == "local":
        aws_s3_data_root = ""
    if data_source == "AWS_S3":
        aws_s3_data_root = aws_s3_data_root.rstrip("/")

    if data_source == "local":
        local_filepath = Path(data_root) / relative_filepath
        return local_filepath

    bucket_name, s3_root_prefix = parse_s3_uri(s3_uri=aws_s3_data_root)
    local_filepath = Path(data_root) / relative_filepath
    if s3_root_prefix != "":
        local_filepath = Path(data_root) / s3_root_prefix / relative_filepath
    s3_key = build_s3_key(
        root_prefix=s3_root_prefix,
        relative_filepath=relative_filepath,
    )

    if not force and local_filepath.exists():
        return local_filepath

    if s3_client is None:
        s3_client = build_s3_client()

    return download_file(
        s3_client=s3_client,
        bucket_name=bucket_name,
        s3_key=s3_key,
        local_filepath=local_filepath,
    )
