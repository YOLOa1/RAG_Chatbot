# RAG Chatbot with Streamlit

This is a full RAG (Retrieval-Augmented Generation) chatbot pipeline implemented using Streamlit and Cohere.

## Features

-   **Document Upload**: Upload multiple PDF documents.
-   **Document Processing**: Automatically splits and embeds document text.
-   **Vector Store**: Uses FAISS for efficient similarity search.
-   **Conversational Interface**: Chat with your documents using a familiar chat interface.
-   **Memory**: Maintains context of the conversation.

## Prerequisites

-   Python 3.8+
-   Cohere API Key

## Installation

1.  Navigate to the project directory:
    ```bash
    cd RAG_Chatbot
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Set up your environment variables:
    -   Rename `.env.example` to `.env`.
    -   Add your Cohere API Key to the `.env` file:
        ```
        COHERE_API_KEY=...
        ```
    -   Alternatively, you can enter your API key directly in the Streamlit sidebar.

## Usage

1.  Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

2.  Open your browser and go to the URL provided (usually `http://localhost:8501`).

3.  **Sidebar**:
    -   Enter your Cohere API Key (if not set in `.env`).
    -   Upload your PDF documents.
    -   Click "Process" to build the vector store.

4.  **Chat**:
    -   Once processing is complete, start asking questions about your documents in the chat input field.

## Project Structure

-   `app.py`: The main Streamlit application file handling the UI and interaction logic.
-   `utils.py`: Contains helper functions for document loading, splitting, embedding, and chain creation.
-   `requirements.txt`: List of Python dependencies.
-   `.env.example`: Template for environment variables.
