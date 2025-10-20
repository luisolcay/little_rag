import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

class BlobStorageError(Exception):
    """Custom exception for Blob Storage related errors."""
    pass

load_dotenv()

# Later auth would be with managed identity
def get_blob_service_client() -> BlobServiceClient:
    """
    Create a BlobServiceClient for the storage account using AAD authentication. 
    """
    account_url = os.getenv("AZURE_BLOB_ENDPOINT")
    if not account_url:
        raise BlobStorageError("AZURE_BLOB_ENDPOINT not set in environment")
    credential = DefaultAzureCredential()
    try:
        return BlobServiceClient(account_url=account_url, credential=credential)
    except Exception as e:
        raise BlobStorageError(f"Failed to create BlobServiceClient: {e}")

def list_blobs_in_container(container_name: str, prefix: str = "") -> list[str]:
    """
    List blob names in a given container, optionally filtering by prefix.
    Returns a list of blob names.
    """
    svc = get_blob_service_client()
    try:
        container = svc.get_container_client(container_name)
    except Exception as e:
        raise BlobStorageError(f"Failed to get container client for '{container_name}': {e}")

    try:
        blob_list = container.list_blobs(name_starts_with=prefix)
        return [blob.name for blob in blob_list]
    except Exception as e:
        raise BlobStorageError(f"Error listing blobs in container '{container_name}' with prefix '{prefix}': {e}")

def upload_blob_from_bytes(container_name: str, blob_name: str, data: bytes, overwrite: bool = True) -> str:
    """Upload bytes data to blob storage and return blob URL."""
    svc = get_blob_service_client()
    container = svc.get_container_client(container_name)
    blob = container.get_blob_client(blob_name)
    
    blob.upload_blob(data, overwrite=overwrite)
    return blob.url

def download_blob_to_file(container_name: str, blob_name: str, download_path: str, overwrite: bool = False) -> None:
    """
    Download a blob from the given container to a local file path.
    If the file already exists and overwrite=False, raises an error.
    """
    svc = get_blob_service_client()
    container = svc.get_container_client(container_name)
    blob = container.get_blob_client(blob_name)

    if os.path.exists(download_path) and not overwrite:
        raise BlobStorageError(f"File '{download_path}' already exists and overwrite=False")

    try:
        stream = blob.download_blob()
        data = stream.readall()
    except Exception as e:
        raise BlobStorageError(f"Error downloading blob '{blob_name}' from container '{container_name}': {e}")

    # Ensure directory exists
    os.makedirs(os.path.dirname(download_path), exist_ok=True)

    try:
        with open(download_path, "wb") as f:
            f.write(data)
    except Exception as e:
        raise BlobStorageError(f"Error writing downloaded blob to file '{download_path}': {e}")

if __name__ == "__main__":
    container = os.getenv("AZURE_BLOB_CONTAINER") or "orbe"
    try:
        blobs = list_blobs_in_container(container)
        print("Blobs in container:", blobs)
    except BlobStorageError as e:
        print("Error listing blobs:", e)
        blobs = []


    for i in range(len(blobs)):
        blob = blobs[i]
        safe_name = blob.replace("/", "_")
        local_path = os.path.join("downloaded_files", f"downloaded_{safe_name}")
        try:
            download_blob_to_file(container, blob, local_path, overwrite=True)
            print(f"Downloaded blob '{blob}' to '{local_path}'")
        except BlobStorageError as e:
            print("Error downloading blob:", e)