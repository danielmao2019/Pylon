"""
DATA.STORAGES API
"""

from data.storages import aws_s3, firestore
from data.storages.storage_router import (
    fetch_file_from_data_source,
    resolve_local_data_root_from_data_source,
)

__all__ = [
    "aws_s3",
    "firestore",
    "fetch_file_from_data_source",
    "resolve_local_data_root_from_data_source",
]
