from typing import List

from google.cloud import firestore


def build_firestore_client(project_id: str) -> firestore.Client:
    # Input validations
    assert isinstance(project_id, str), f"{type(project_id)=}"
    assert project_id != "", f"{project_id=}"

    return firestore.Client(project=project_id)


def list_collection_ids(firestore_client: firestore.Client) -> List[str]:
    # Input validations
    assert isinstance(firestore_client, firestore.Client), f"{type(firestore_client)=}"

    collection_ids = [collection.id for collection in firestore_client.collections()]
    assert isinstance(collection_ids, list), f"{type(collection_ids)=}"
    return sorted(collection_ids)


def assert_required_collections_exist(
    firestore_client: firestore.Client,
    models_collection: str,
    users_collection: str,
) -> None:
    # Input validations
    assert isinstance(firestore_client, firestore.Client), f"{type(firestore_client)=}"
    assert isinstance(models_collection, str), f"{type(models_collection)=}"
    assert models_collection != "", f"{models_collection=}"
    assert isinstance(users_collection, str), f"{type(users_collection)=}"
    assert users_collection != "", f"{users_collection=}"

    collection_ids = list_collection_ids(firestore_client=firestore_client)
    assert (
        models_collection in collection_ids
    ), f"{models_collection=}, {collection_ids=}"
    assert users_collection in collection_ids, f"{users_collection=}, {collection_ids=}"


def list_document_ids(
    firestore_client: firestore.Client,
    collection_name: str,
) -> List[str]:
    # Input validations
    assert isinstance(firestore_client, firestore.Client), f"{type(firestore_client)=}"
    assert isinstance(collection_name, str), f"{type(collection_name)=}"
    assert collection_name != "", f"{collection_name=}"

    collection = firestore_client.collection(collection_name)
    snapshots = list(collection.stream())
    document_ids = [snapshot.id for snapshot in snapshots]
    return sorted(document_ids)
