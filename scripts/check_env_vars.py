"""
Environment Variables Diagnostic Script
======================================

This script checks if environment variables are being loaded correctly
from the .env file.
"""

import os
from dotenv import load_dotenv

def check_environment_variables():
    """Check if environment variables are loaded correctly."""
    
    print("🔍 ENVIRONMENT VARIABLES DIAGNOSTIC")
    print("=" * 50)
    
    # Load .env file
    print("📁 Loading .env file...")
    load_result = load_dotenv()
    print(f"✅ load_dotenv() result: {load_result}")
    
    # Check if .env file exists
    env_file = Path('.env')
    if env_file.exists():
        print(f"✅ .env file exists: {env_file.absolute()}")
        print(f"📊 .env file size: {env_file.stat().st_size} bytes")
    else:
        print("❌ .env file not found")
        return
    
    # List of required variables
    required_vars = [
        'AZURE_OPENAI_ENDPOINT',
        'AZURE_OPENAI_API_KEY',
        'AZURE_AI_SEARCH_ENDPOINT',
        'AZURE_SEARCH_API_KEY_ADMIN',
        'AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT',
        'AZURE_DOCUMENT_INTELLIGENCE_API_KEY',
        'AZURE_COSMOS_DB_ENDPOINT',
        'AZURE_COSMOS_DB_PRIMARY_KEY',
        'AZURE_REDIS_HOST',
        'AZURE_REDIS_PRIMARY_KEY'
    ]
    
    print(f"\n🔍 Checking {len(required_vars)} required variables:")
    print("-" * 50)
    
    missing_vars = []
    present_vars = []
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Show first 20 characters and last 4 characters for security
            masked_value = f"{value[:20]}...{value[-4:]}" if len(value) > 24 else f"{value[:10]}..." if len(value) > 10 else "***"
            print(f"✅ {var}: {masked_value}")
            present_vars.append(var)
        else:
            print(f"❌ {var}: NOT SET")
            missing_vars.append(var)
    
    print(f"\n📊 SUMMARY:")
    print(f"✅ Variables present: {len(present_vars)}")
    print(f"❌ Variables missing: {len(missing_vars)}")
    
    if missing_vars:
        print(f"\n❌ Missing variables:")
        for var in missing_vars:
            print(f"  - {var}")
    else:
        print(f"\n🎉 All required variables are present!")
    
    # Check some optional variables
    optional_vars = [
        'AZURE_OPENAI_EMBEDDING_DEPLOYMENT',
        'AZURE_OPENAI_CHAT_DEPLOYMENT',
        'AZURE_COSMOS_DB_DATABASE_NAME',
        'AZURE_COSMOS_DB_CONTAINER_NAME'
    ]
    
    print(f"\n🔍 Checking optional variables:")
    print("-" * 30)
    
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {value}")
        else:
            print(f"⚠️ {var}: NOT SET (optional)")
    
    # Test specific Azure services
    print(f"\n🧪 Testing Azure service configurations:")
    print("-" * 40)
    
    # Azure OpenAI
    openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    openai_key = os.getenv('AZURE_OPENAI_API_KEY')
    if openai_endpoint and openai_key:
        print(f"✅ Azure OpenAI: Configured")
        print(f"   Endpoint: {openai_endpoint}")
    else:
        print(f"❌ Azure OpenAI: Not configured")
    
    # Azure AI Search
    search_endpoint = os.getenv('AZURE_AI_SEARCH_ENDPOINT')
    search_key = os.getenv('AZURE_SEARCH_API_KEY_ADMIN')
    if search_endpoint and search_key:
        print(f"✅ Azure AI Search: Configured")
        print(f"   Endpoint: {search_endpoint}")
    else:
        print(f"❌ Azure AI Search: Not configured")
    
    # Azure Document Intelligence
    doc_intel_endpoint = os.getenv('AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT')
    doc_intel_key = os.getenv('AZURE_DOCUMENT_INTELLIGENCE_API_KEY')
    if doc_intel_endpoint and doc_intel_key:
        print(f"✅ Azure Document Intelligence: Configured")
        print(f"   Endpoint: {doc_intel_endpoint}")
    else:
        print(f"❌ Azure Document Intelligence: Not configured")
    
    # Azure Cosmos DB
    cosmos_endpoint = os.getenv('AZURE_COSMOS_DB_ENDPOINT')
    cosmos_key = os.getenv('AZURE_COSMOS_DB_PRIMARY_KEY')
    if cosmos_endpoint and cosmos_key:
        print(f"✅ Azure Cosmos DB: Configured")
        print(f"   Endpoint: {cosmos_endpoint}")
    else:
        print(f"❌ Azure Cosmos DB: Not configured")
    
    # Azure Redis
    redis_host = os.getenv('AZURE_REDIS_HOST')
    redis_key = os.getenv('AZURE_REDIS_PRIMARY_KEY')
    if redis_host and redis_key:
        print(f"✅ Azure Redis: Configured")
        print(f"   Host: {redis_host}")
    else:
        print(f"❌ Azure Redis: Not configured")
    
    return len(missing_vars) == 0

if __name__ == "__main__":
    from pathlib import Path
    success = check_environment_variables()
    
    if success:
        print(f"\n🎉 Environment is ready for Azure RAG testing!")
    else:
        print(f"\n⚠️ Please check your .env file configuration.")
