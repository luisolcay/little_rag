import streamlit as st
from api_utils import get_api_response

def display_citations(citations):
    """Display citations panel."""
    if not citations or len(citations) == 0:
        return
    
    with st.expander("📚 Fuentes consultadas", expanded=True):
        for i, citation in enumerate(citations, 1):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**[{i}] {citation['document_name']}**")
                if citation.get('page_number'):
                    st.caption(f"📄 Página {citation['page_number']}")
                if citation.get('section'):
                    st.caption(f"📑 Sección: {citation['section']}")
            
            with col2:
                relevance = citation.get('relevance_score', 0)
                st.metric("Relevancia", f"{relevance:.0%}")
            
            if citation.get('content_snippet'):
                with st.expander("Ver extracto"):
                    st.text(citation['content_snippet'])
            
            if i < len(citations):
                st.divider()

def display_chat_interface():
    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show citations if available
            if message["role"] == "assistant" and message.get("citations"):
                display_citations(message["citations"])

    if prompt := st.chat_input("Query:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Generating response..."):
            use_context = st.session_state.get('use_context', True)
            response = get_api_response(
                prompt,
                st.session_state.session_id,
                st.session_state.model,
                use_context
            )
            
            if response:
                st.session_state.session_id = response.get('session_id')
                
                message_data = {
                    "role": "assistant",
                    "content": response['answer'],
                    "citations": response.get('citations', [])
                }
                st.session_state.messages.append(message_data)
                
                with st.chat_message("assistant"):
                    st.markdown(response['answer'])
                    
                    # Display citations
                    if response.get('citations'):
                        display_citations(response['citations'])
                    
                    with st.expander("Details"):
                        st.subheader("Generated Answer")
                        st.code(response['answer'])
                        st.subheader("Model Used")
                        st.code(response['model'])
                        st.subheader("Session ID")
                        st.code(response['session_id'])
                        if response.get('citations'):
                            st.subheader("Citations")
                            st.json(response['citations'])
            else:
                st.error("Failed to get a response from the API. Please try again.")