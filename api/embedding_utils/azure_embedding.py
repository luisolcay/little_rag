import os
import httpx
from dotenv import load_dotenv

load_dotenv()

ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT") 
API_VER = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

async def test_embedding():
    url = f"{ENDPOINT}/openai/deployments/{DEPLOYMENT}/embeddings?api-version={API_VER}"
    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY
    }
    body = {
        "input": ["Hello world", "Test embedding"]
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, headers=headers, json=body)
        print("Status:", resp.status_code)
        try:
            j = resp.json()
        except Exception as e:
            print("No JSON:", resp.text)
            raise
        print("Response JSON:", j)
        
        if "data" in j:
            for item in j["data"]:
                print("Embedding vector length:", len(item.get("embedding", [])))

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_embedding())