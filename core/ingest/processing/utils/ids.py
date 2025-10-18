import json, hashlib

def deterministic_chunk_id(metadata: dict) -> str:
    """
    Generates a stable SHA1 ID from core metadata fields.
    """
    base = json.dumps(
        {
            "document_id": metadata.get("document_id"),
            "document_blob": metadata.get("document_blob"),
            "page_number": metadata.get("page_number"),
            "chunk_index": metadata.get("chunk_index"),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha1(base.encode("utf-8")).hexdigest()