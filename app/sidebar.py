import streamlit as st
from api_utils import upload_document, list_documents, delete_document, get_documents_list, get_index_status

def display_sidebar():
    # RAG Configuration
    st.sidebar.header("⚙️ Configuración RAG")
    use_context = st.sidebar.checkbox(
        "Usar documentos como contexto",
        value=True,
        help="Desactiva para respuestas sin documentos"
    )
    st.session_state.use_context = use_context
    
    # Model Selection
    model_options = ["gpt-4o", "gpt-4o-mini"]
    st.sidebar.selectbox("Modelo", options=model_options, key="model")
    
    st.sidebar.divider()
    
    # Upload Document
    st.sidebar.header("📤 Subir Documento")
    uploaded_file = st.sidebar.file_uploader(
        "Selecciona archivo",
        type=["pdf", "docx", "html", "txt"],
        help="Se subirá a Azure Blob Storage"
    )
    
    if uploaded_file is not None:
        if st.sidebar.button("📥 Subir e Indexar"):
            with st.spinner("Procesando (Blob Storage + OCR + Indexación)..."):
                upload_response = upload_document(uploaded_file)
                
                if upload_response:
                    if upload_response.get('indexing_result'):
                        st.sidebar.success(f"✅ {uploaded_file.name}")
                        
                        # Show details
                        col1, col2 = st.sidebar.columns(2)
                        with col1:
                            st.metric("Chunks", upload_response['processing_result']['chunks_count'])
                        with col2:
                            ocr_used = "✓" if upload_response.get('needs_ocr') else "✗"
                            st.metric("OCR", ocr_used)
                        
                        indexing = upload_response['indexing_result']
                        st.sidebar.caption(f"⏱️ {indexing.get('processing_time', 0):.2f}s")
                    else:
                        st.sidebar.warning("⚠️ Procesado pero no indexado")
    
    st.sidebar.divider()
    
    # Documents from Cosmos DB
    st.sidebar.header("📄 Documentos")
    
    if st.sidebar.button("🔄 Actualizar"):
        st.session_state.documents_list = get_documents_list()
    
    if 'documents_list' not in st.session_state:
        st.session_state.documents_list = get_documents_list()
    
    docs_data = st.session_state.get('documents_list')
    if docs_data and docs_data.get('documents'):
        st.sidebar.metric("Total", docs_data['total'])
        
        with st.sidebar.expander("Ver lista"):
            for doc in docs_data['documents'][:10]:  # Show first 10
                st.text(f"📄 {doc['filename']}")
                st.caption(f"Chunks: {doc['chunks_count']} | OCR: {'Sí' if doc['needs_ocr'] else 'No'}")
                st.caption(f"🕒 {doc['upload_timestamp'][:10]}")
                st.divider()
    
    st.sidebar.divider()
    
    # Azure AI Search Status
    st.sidebar.header("🔍 Azure AI Search")
    
    if 'index_status' not in st.session_state:
        st.session_state.index_status = get_index_status()
    
    status = st.session_state.get('index_status')
    if status and status.get('index_exists'):
        stats = status.get('statistics', {})
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Docs", stats.get('document_count', 0))
        with col2:
            st.metric("Chunks", stats.get('storage_size', 0))