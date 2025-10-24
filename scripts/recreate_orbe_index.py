"""
Script to recreate the orbe-documents index with the correct schema
===================================================================
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(override=True)

from core.vector.azure_search.index_manager import IndexManager

def recreate_index():
    """Recreate the orbe-documents index with correct schema."""
    print("🔄 Recreating orbe-documents index...")
    print("=" * 60)
    
    index_manager = IndexManager(index_name="orbe-documents")
    
    # Check if index exists
    if index_manager.index_exists():
        print("⚠️ Index already exists. Deleting it...")
        index_manager.delete_index()
        print("✅ Index deleted")
    
    # Create new index
    print("\n📋 Creating new index with correct schema...")
    success = index_manager.create_index()
    
    if success:
        print("✅ Index 'orbe-documents' created successfully!")
        
        # Get index info
        print("\n📊 Index Information:")
        info = index_manager.get_index_info()
        if info:
            print(f"  Name: {info['name']}")
            print(f"  Fields: {info['field_count']}")
            print(f"  Vector search: {info['vector_search_enabled']}")
            print(f"  Semantic search: {info['semantic_search_enabled']}")
            
            print("\n  Fields:")
            for field in info['fields']:
                print(f"    - {field['name']}: {field['type']}")
                print(f"      searchable={field['searchable']}, filterable={field['filterable']}")
    else:
        print("❌ Failed to create index")
    
    print("\n✅ Done!")

if __name__ == "__main__":
    recreate_index()

