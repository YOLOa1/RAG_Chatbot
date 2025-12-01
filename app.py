import streamlit as st
import os
from dotenv import load_dotenv
from utils import process_documents, create_vectorstore, get_conversation_chain

def main():
    load_dotenv()
    st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

    st.header("🤖 RAG Chatbot with Cohere")

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    # Sidebar for configuration and file upload
    with st.sidebar:
        st.subheader("Configuration")
        api_key = st.text_input("Cohere API Key", type="password")
        if api_key:
            os.environ["COHERE_API_KEY"] = api_key
        
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        
        if st.button("Process"):
            if not api_key and not os.getenv("COHERE_API_KEY"):
                st.error("Please provide a Cohere API Key.")
            elif not pdf_docs:
                st.error("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing documents..."):
                    # Get PDF text
                    raw_documents = process_documents(pdf_docs)
                    
                    # Create vector store
                    vectorstore = create_vectorstore(raw_documents)
                    
                    # Create conversation chain
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    
                    st.session_state.processComplete = True
                    st.success("Documents processed successfully!")

    # Chat Interface
    if st.session_state.processComplete:
        # Display chat messages from history on app rerun
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # React to user input
        if prompt := st.chat_input("Ask a question about your documents:"):
            # Display user message in chat message container
            st.chat_message("user").markdown(prompt)
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            with st.spinner("Thinking..."):
                response = st.session_state.conversation({"question": prompt})
                answer = response['answer']
                
                # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    st.markdown(answer)
                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
    else:
        st.info("Please upload documents and click 'Process' to start chatting.")

if __name__ == '__main__':
    main()
