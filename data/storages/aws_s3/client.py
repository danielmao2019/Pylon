import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import boto3
from botocore.config import Config

AWS_S3_CONFIG_FILENAME = "config.local.json"


def get_aws_s3_config_filepath() -> Path:
    return Path(__file__).resolve().parent / AWS_S3_CONFIG_FILENAME


def load_aws_s3_config() -> Dict[str, Any]:
    # Input normalizations
    config_filepath = get_aws_s3_config_filepath()
    assert config_filepath.exists(), f"Missing AWS S3 config file: {config_filepath}"
    config_data = json.loads(config_filepath.read_text(encoding="utf-8"))
    assert isinstance(config_data, dict), f"{type(config_data)=}"
    assert "aws_region" in config_data, f"{config_data=}"
    assert "max_attempts" in config_data, f"{config_data=}"
    assert "aws_access_key_id" in config_data, f"{config_data=}"
    assert "aws_secret_access_key" in config_data, f"{config_data=}"

    aws_region = config_data["aws_region"]
    max_attempts = config_data["max_attempts"]
    aws_access_key_id = config_data["aws_access_key_id"]
    aws_secret_access_key = config_data["aws_secret_access_key"]
    aws_session_token = None
    if "aws_session_token" in config_data:
        aws_session_token = config_data["aws_session_token"]
    assert isinstance(aws_region, str), f"{type(aws_region)=}"
    assert aws_region != "", f"{aws_region=}"
    assert isinstance(max_attempts, int), f"{type(max_attempts)=}"
    assert max_attempts > 0, f"{max_attempts=}"
    assert isinstance(aws_access_key_id, str), f"{type(aws_access_key_id)=}"
    assert aws_access_key_id != "", f"{aws_access_key_id=}"
    assert isinstance(aws_secret_access_key, str), f"{type(aws_secret_access_key)=}"
    assert aws_secret_access_key != "", f"{aws_secret_access_key=}"
    assert aws_session_token is None or isinstance(
        aws_session_token, str
    ), f"{type(aws_session_token)=}"
    assert aws_session_token is None or aws_session_token != "", f"{aws_session_token=}"

    loaded_config = {
        "aws_region": aws_region,
        "max_attempts": max_attempts,
        "aws_access_key_id": aws_access_key_id,
        "aws_secret_access_key": aws_secret_access_key,
    }
    if aws_session_token is not None:
        loaded_config["aws_session_token"] = aws_session_token
    return loaded_config


def build_s3_client() -> Any:
    s3_config = load_aws_s3_config()

    aws_config = Config(
        retries={
            "max_attempts": s3_config["max_attempts"],
            "mode": "standard",
        }
    )
    if "aws_session_token" in s3_config:
        return boto3.client(
            service_name="s3",
            region_name=s3_config["aws_region"],
            aws_access_key_id=s3_config["aws_access_key_id"],
            aws_secret_access_key=s3_config["aws_secret_access_key"],
            aws_session_token=s3_config["aws_session_token"],
            config=aws_config,
        )
    return boto3.client(
        service_name="s3",
        region_name=s3_config["aws_region"],
        aws_access_key_id=s3_config["aws_access_key_id"],
        aws_secret_access_key=s3_config["aws_secret_access_key"],
        config=aws_config,
    )


def parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    # Input validations
    assert isinstance(s3_uri, str), f"{type(s3_uri)=}"
    assert s3_uri.startswith("s3://"), f"{s3_uri=}"

    # Input normalizations
    uri_without_scheme = s3_uri[len("s3://") :].strip()
    assert uri_without_scheme != "", f"{s3_uri=}"
    if uri_without_scheme.endswith("/"):
        uri_without_scheme = uri_without_scheme[:-1]

    if "/" in uri_without_scheme:
        bucket_name, root_prefix = uri_without_scheme.split("/", 1)
    else:
        bucket_name, root_prefix = uri_without_scheme, ""

    assert bucket_name != "", f"{s3_uri=}"
    return bucket_name, root_prefix


def build_s3_key(root_prefix: str, relative_filepath: str) -> str:
    # Input validations
    assert isinstance(root_prefix, str), f"{type(root_prefix)=}"
    assert isinstance(relative_filepath, str), f"{type(relative_filepath)=}"
    assert relative_filepath != "", f"{relative_filepath=}"

    # Input normalizations
    normalized_root_prefix = root_prefix.strip("/")
    normalized_relative_filepath = relative_filepath.lstrip("/")
    assert normalized_relative_filepath != "", f"{relative_filepath=}"

    if normalized_root_prefix == "":
        return normalized_relative_filepath
    return f"{normalized_root_prefix}/{normalized_relative_filepath}"


def list_child_prefix_names(
    s3_client: Any,
    bucket_name: str,
    root_prefix: str,
) -> List[str]:
    # Input validations
    assert s3_client is not None, f"{s3_client=}"
    assert isinstance(bucket_name, str), f"{type(bucket_name)=}"
    assert bucket_name != "", f"{bucket_name=}"
    assert isinstance(root_prefix, str), f"{type(root_prefix)=}"

    normalized_root_prefix = root_prefix.strip("/")
    if normalized_root_prefix != "":
        normalized_root_prefix = f"{normalized_root_prefix}/"

    response: Dict[str, Any] = s3_client.list_objects_v2(
        Bucket=bucket_name,
        Prefix=normalized_root_prefix,
        Delimiter="/",
    )
    assert "CommonPrefixes" in response, f"{response=}"
    common_prefixes = response["CommonPrefixes"]
    assert isinstance(common_prefixes, list), f"{type(common_prefixes)=}"

    children: List[str] = []
    for prefix_entry in common_prefixes:
        assert isinstance(prefix_entry, dict), f"{type(prefix_entry)=}"
        assert "Prefix" in prefix_entry, f"{prefix_entry=}"
        child_prefix = prefix_entry["Prefix"]
        assert isinstance(child_prefix, str), f"{type(child_prefix)=}"
        assert child_prefix.endswith("/"), f"{child_prefix=}"
        if normalized_root_prefix != "":
            assert child_prefix.startswith(
                normalized_root_prefix
            ), f"{child_prefix=}, {normalized_root_prefix=}"
            child_name = child_prefix[len(normalized_root_prefix) : -1]
        else:
            child_name = child_prefix[:-1]
        assert child_name != "", f"{child_prefix=}"
        children.append(child_name)
    return sorted(children)


def download_file(
    s3_client: Any,
    bucket_name: str,
    s3_key: str,
    local_filepath: Path,
) -> Path:
    # Input validations
    assert s3_client is not None, f"{s3_client=}"
    assert isinstance(bucket_name, str), f"{type(bucket_name)=}"
    assert bucket_name != "", f"{bucket_name=}"
    assert isinstance(s3_key, str), f"{type(s3_key)=}"
    assert s3_key != "", f"{s3_key=}"
    assert isinstance(local_filepath, Path), f"{type(local_filepath)=}"

    local_filepath.parent.mkdir(parents=True, exist_ok=True)
    s3_client.download_file(
        Bucket=bucket_name,
        Key=s3_key,
        Filename=str(local_filepath),
    )
    assert local_filepath.exists(), f"{local_filepath=}"
    return local_filepath
