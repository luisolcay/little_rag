import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

class BlobStorageError(Exception):
    pass

# Cargar variables de entorno desde .env
load_dotenv()

def get_blob_service_client() -> BlobServiceClient:
 
    account_url = os.getenv("AZURE_BLOB_ACCOUNT_URL")
    if not account_url:
        raise BlobStorageError("AZURE_BLOB_ACCOUNT_URL not set in environment")
    credential = DefaultAzureCredential()
    return BlobServiceClient(account_url=account_url, credential=credential)

def list_blobs_in_container(container_name: str, prefix: str = "") -> list[str]:
    
    svc = get_blob_service_client()
    container = svc.get_container_client(container_name)
    blob_list = container.list_blobs(name_starts_with=prefix)
    return [blob.name for blob in blob_list]

def download_blob_to_file(
    container_name: str,
    blob_name: str,
    download_path: str,
    overwrite: bool = False
) -> None:
   
    svc = get_blob_service_client()
    container = svc.get_container_client(container_name)
    blob = container.get_blob_client(blob_name)

    if os.path.exists(download_path) and not overwrite:
        raise BlobStorageError(f"File {download_path} already exists and overwrite=False")

    stream = blob.download_blob()
    data = stream.readall()

    with open(download_path, "wb") as f:
        f.write(data)

if __name__ == "__main__":
    # Para pruebas locales
    container = os.getenv("AZURE_BLOB_CONTAINER") or "orbe"
    blobs = []
    try:
        blobs = list_blobs_in_container(container)
        print("Blobs in container:", blobs)
    except BlobStorageError as e:
        print("Error listing blobs:", e)

    if blobs:
        first = blobs[0]
        safe_name = first.replace("/", "_")
        local_path = f"./downloaded_{safe_name}"
        try:
            download_blob_to_file(container, first, local_path, overwrite=True)
            print(f"Downloaded blob {first} to {local_path}")
        except BlobStorageError as e:
            print("Error downloading blob:", e)