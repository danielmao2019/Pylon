import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

AWS_S3_CONFIG_FILENAME = "config.local.json"


def get_aws_s3_config_filepath() -> Path:
    """Return the local AWS S3 config filepath.

    Args:
        None.

    Returns:
        Path to the local AWS S3 config JSON file.
    """
    return Path(__file__).resolve().parent / AWS_S3_CONFIG_FILENAME


def _validate_aws_s3_config_data(config_data: Any) -> None:
    assert isinstance(
        config_data, dict
    ), f"Expected AWS S3 config payload to be a dictionary. {type(config_data)=}"
    assert (
        "aws_region" in config_data
    ), f"Expected AWS S3 config to define `aws_region`. {config_data.keys()=}"
    assert (
        "max_attempts" in config_data
    ), f"Expected AWS S3 config to define `max_attempts`. {config_data.keys()=}"
    assert (
        "aws_access_key_id" in config_data
    ), f"Expected AWS S3 config to define `aws_access_key_id`. {config_data.keys()=}"
    assert "aws_secret_access_key" in config_data, (
        "Expected AWS S3 config to define `aws_secret_access_key`. "
        f"{config_data.keys()=}"
    )
    assert isinstance(config_data["aws_region"], str), (
        "Expected AWS S3 config `aws_region` to be a string. "
        f"{type(config_data['aws_region'])=}"
    )
    assert (
        config_data["aws_region"] != ""
    ), f"Expected AWS S3 config `aws_region` to be non-empty. {config_data['aws_region']=}"
    assert isinstance(config_data["max_attempts"], int), (
        "Expected AWS S3 config `max_attempts` to be an integer. "
        f"{type(config_data['max_attempts'])=}"
    )
    assert config_data["max_attempts"] > 0, (
        "Expected AWS S3 config `max_attempts` to be positive. "
        f"{config_data['max_attempts']=}"
    )
    assert isinstance(config_data["aws_access_key_id"], str), (
        "Expected AWS S3 config `aws_access_key_id` to be a string. "
        f"{type(config_data['aws_access_key_id'])=}"
    )
    assert config_data["aws_access_key_id"] != "", (
        "Expected AWS S3 config `aws_access_key_id` to be non-empty. "
        f"{config_data['aws_access_key_id']=}"
    )
    assert isinstance(config_data["aws_secret_access_key"], str), (
        "Expected AWS S3 config `aws_secret_access_key` to be a string. "
        f"{type(config_data['aws_secret_access_key'])=}"
    )
    assert config_data["aws_secret_access_key"] != "", (
        "Expected AWS S3 config `aws_secret_access_key` to be non-empty. "
        f"{config_data['aws_secret_access_key']=}"
    )
    if "aws_session_token" in config_data:
        assert isinstance(config_data["aws_session_token"], str), (
            "Expected AWS S3 config `aws_session_token` to be a string when present. "
            f"{type(config_data['aws_session_token'])=}"
        )
        assert config_data["aws_session_token"] != "", (
            "Expected AWS S3 config `aws_session_token` to be non-empty when present. "
            f"{config_data['aws_session_token']=}"
        )


def load_aws_s3_config() -> Dict[str, Any]:
    """Load and validate local AWS S3 configuration.

    Args:
        None.

    Returns:
        AWS S3 configuration dictionary.
    """
    config_filepath = get_aws_s3_config_filepath()
    assert config_filepath.exists(), (
        "Expected AWS S3 config file to exist. " f"{config_filepath=}"
    )
    config_data = json.loads(config_filepath.read_text(encoding="utf-8"))
    _validate_aws_s3_config_data(config_data=config_data)

    aws_region = config_data["aws_region"]
    max_attempts = config_data["max_attempts"]
    aws_access_key_id = config_data["aws_access_key_id"]
    aws_secret_access_key = config_data["aws_secret_access_key"]
    aws_session_token = None
    if "aws_session_token" in config_data:
        aws_session_token = config_data["aws_session_token"]

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
    """Build a boto3 S3 client from local config.

    Args:
        None.

    Returns:
        Configured boto3 S3 client.
    """
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
    """Parse an S3 URI into bucket and prefix.

    Args:
        s3_uri: S3 URI in `s3://bucket/prefix` format.

    Returns:
        Tuple of bucket name and root prefix.
    """

    def _validate_inputs() -> None:
        assert isinstance(s3_uri, str), (
            "Expected `s3_uri` to be a string. " f"{type(s3_uri)=}"
        )
        assert s3_uri.startswith("s3://"), (
            "Expected `s3_uri` to start with `s3://`. " f"{s3_uri=}"
        )
        assert s3_uri[len("s3://") :].split("/", 1)[0] != "", (
            "Expected `s3_uri` to include a non-empty bucket name. " f"{s3_uri=}"
        )
        assert not s3_uri[len("s3://") :].split("/", 1)[0].isspace(), (
            "Expected `s3_uri` to include a non-empty bucket name. " f"{s3_uri=}"
        )

    _validate_inputs()

    def _normalize_inputs(s3_uri: str) -> str:
        s3_uri = s3_uri.strip()
        if s3_uri.endswith("/"):
            s3_uri = s3_uri[:-1]
        assert s3_uri.startswith("s3://"), f"{s3_uri=}"
        return s3_uri

    s3_uri = _normalize_inputs(s3_uri=s3_uri)

    uri_without_scheme = s3_uri[len("s3://") :]

    if "/" in uri_without_scheme:
        bucket_name, root_prefix = uri_without_scheme.split("/", 1)
    else:
        bucket_name, root_prefix = uri_without_scheme, ""

    return bucket_name, root_prefix


def build_s3_key(root_prefix: str, relative_filepath: str) -> str:
    """Build an S3 object key from a root prefix and relative path.

    Args:
        root_prefix: S3 prefix that owns the relative file.
        relative_filepath: File path relative to the S3 root prefix.

    Returns:
        S3 object key.
    """

    def _validate_inputs() -> None:
        assert isinstance(root_prefix, str), (
            "Expected `root_prefix` to be a string. " f"{type(root_prefix)=}"
        )

        assert isinstance(relative_filepath, str), (
            "Expected `relative_filepath` to be a string. "
            f"{type(relative_filepath)=}"
        )
        assert relative_filepath != "", (
            "Expected `relative_filepath` to be non-empty. " f"{relative_filepath=}"
        )
        assert any(part != "" for part in relative_filepath.split("/")), (
            "Expected `relative_filepath` to contain a non-slash path segment. "
            f"{relative_filepath=}"
        )

    _validate_inputs()

    def _normalize_inputs(root_prefix: str, relative_filepath: str) -> Tuple[str, str]:
        root_prefix = root_prefix.strip("/")
        relative_filepath = relative_filepath.lstrip("/")
        assert relative_filepath != "", f"{relative_filepath=}"
        return root_prefix, relative_filepath

    root_prefix, relative_filepath = _normalize_inputs(
        root_prefix=root_prefix,
        relative_filepath=relative_filepath,
    )

    if root_prefix == "":
        return relative_filepath
    return f"{root_prefix}/{relative_filepath}"


def list_child_prefix_names(
    s3_client: Any,
    bucket_name: str,
    root_prefix: str,
) -> List[str]:
    """List direct child prefix names under an S3 prefix.

    Args:
        s3_client: Boto3 S3 client.
        bucket_name: S3 bucket name.
        root_prefix: Parent prefix inside the bucket.

    Returns:
        Sorted child prefix names.
    """

    def _validate_inputs() -> None:
        assert s3_client is not None, (
            "Expected `s3_client` to be initialized. " f"{s3_client=}"
        )

        assert isinstance(bucket_name, str), (
            "Expected `bucket_name` to be a string. " f"{type(bucket_name)=}"
        )
        assert bucket_name != "", (
            "Expected `bucket_name` to be non-empty. " f"{bucket_name=}"
        )

        assert isinstance(root_prefix, str), (
            "Expected `root_prefix` to be a string. " f"{type(root_prefix)=}"
        )

    _validate_inputs()

    def _normalize_inputs(root_prefix: str) -> str:
        root_prefix = root_prefix.strip("/")
        if root_prefix != "":
            root_prefix = f"{root_prefix}/"
        return root_prefix

    root_prefix = _normalize_inputs(root_prefix=root_prefix)

    response: Dict[str, Any] = s3_client.list_objects_v2(
        Bucket=bucket_name,
        Prefix=root_prefix,
        Delimiter="/",
    )
    assert (
        "CommonPrefixes" in response
    ), f"Expected S3 list response to include `CommonPrefixes`. {response=}"
    common_prefixes = response["CommonPrefixes"]
    assert isinstance(common_prefixes, list), (
        "Expected S3 list response `CommonPrefixes` to be a list. "
        f"{type(common_prefixes)=}"
    )

    children: List[str] = []
    for prefix_entry in common_prefixes:
        assert isinstance(prefix_entry, dict), (
            "Expected each S3 common-prefix entry to be a dictionary. "
            f"{type(prefix_entry)=}"
        )
        assert (
            "Prefix" in prefix_entry
        ), f"Expected S3 common-prefix entry to include `Prefix`. {prefix_entry=}"
        child_prefix = prefix_entry["Prefix"]
        assert isinstance(child_prefix, str), (
            "Expected S3 common-prefix `Prefix` value to be a string. "
            f"{type(child_prefix)=}"
        )
        assert child_prefix.endswith("/"), (
            "Expected S3 common-prefix `Prefix` value to end with `/`. "
            f"{child_prefix=}"
        )
        if root_prefix != "":
            assert child_prefix.startswith(root_prefix), (
                "Expected child prefix to be below normalized root prefix. "
                f"{child_prefix=} {root_prefix=}"
            )
            child_name = child_prefix[len(root_prefix) : -1]
        else:
            child_name = child_prefix[:-1]
        assert child_name != "", (
            "Expected S3 child prefix name to be non-empty. " f"{child_prefix=}"
        )
        children.append(child_name)
    return sorted(children)


def download_file(
    s3_client: Any,
    bucket_name: str,
    s3_key: str,
    local_filepath: Path,
) -> Path:
    """Download one S3 object to a local path.

    Args:
        s3_client: Boto3 S3 client.
        bucket_name: S3 bucket name.
        s3_key: S3 object key.
        local_filepath: Local output filepath.

    Returns:
        Local filepath containing the downloaded object.
    """

    def _validate_inputs() -> None:
        assert s3_client is not None, (
            "Expected `s3_client` to be initialized. " f"{s3_client=}"
        )

        assert isinstance(bucket_name, str), (
            "Expected `bucket_name` to be a string. " f"{type(bucket_name)=}"
        )
        assert bucket_name != "", (
            "Expected `bucket_name` to be non-empty. " f"{bucket_name=}"
        )

        assert isinstance(s3_key, str), (
            "Expected `s3_key` to be a string. " f"{type(s3_key)=}"
        )
        assert s3_key != "", f"Expected `s3_key` to be non-empty. {s3_key=}"

        assert isinstance(local_filepath, Path), (
            "Expected `local_filepath` to be a `Path`. " f"{type(local_filepath)=}"
        )

    _validate_inputs()

    local_filepath.parent.mkdir(parents=True, exist_ok=True)
    s3_client.download_file(
        Bucket=bucket_name,
        Key=s3_key,
        Filename=str(local_filepath),
    )
    assert local_filepath.exists(), (
        "Expected S3 download to create the local file. " f"{local_filepath=}"
    )
    return local_filepath


def file_exists(
    s3_client: Any,
    bucket_name: str,
    s3_key: str,
) -> bool:
    """Check whether one S3 object exists.

    Args:
        s3_client: Boto3 S3 client.
        bucket_name: S3 bucket name.
        s3_key: Object key inside the bucket.

    Returns:
        Whether the S3 object exists.
    """

    def _validate_inputs() -> None:
        assert s3_client is not None, (
            "Expected `s3_client` to be initialized. " f"{s3_client=}"
        )

        assert isinstance(bucket_name, str), (
            "Expected `bucket_name` to be a string. " f"{type(bucket_name)=}"
        )
        assert bucket_name != "", (
            "Expected `bucket_name` to be non-empty. " f"{bucket_name=}"
        )

        assert isinstance(s3_key, str), (
            "Expected `s3_key` to be a string. " f"{type(s3_key)=}"
        )
        assert s3_key != "", f"Expected `s3_key` to be non-empty. {s3_key=}"

    _validate_inputs()

    try:
        s3_client.head_object(
            Bucket=bucket_name,
            Key=s3_key,
        )
    except ClientError as exc:

        def _validate_error_response() -> None:
            assert isinstance(exc.response, dict), (
                "Expected boto3 ClientError response to be a dictionary. "
                f"{type(exc.response)=}"
            )
            assert "Error" in exc.response, (
                "Expected boto3 ClientError response to contain `Error`. "
                f"{exc.response=}"
            )
            assert isinstance(exc.response["Error"], dict), (
                "Expected boto3 ClientError `Error` payload to be a dictionary. "
                f"{type(exc.response['Error'])=}"
            )
            assert "Code" in exc.response["Error"], (
                "Expected boto3 ClientError `Error` payload to contain `Code`. "
                f"{exc.response['Error']=}"
            )
            assert isinstance(exc.response["Error"]["Code"], str), (
                "Expected boto3 ClientError code to be a string. "
                f"{type(exc.response['Error']['Code'])=}"
            )

        _validate_error_response()

        response = exc.response
        error_payload = response["Error"]
        error_code = error_payload["Code"]
        if error_code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise
    return True
