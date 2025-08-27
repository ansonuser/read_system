# RAG QA Reader

A Streamlit application for Document Q&A using Retrieval-Augmented Generation (RAG) with LLM.

## Project Description

This project implements a Document Q&A system using RAG with LLM. It allows users to upload or paste document content and ask questions about it. The system uses a local embedder for chunking and vector storage, and LLM for generating answers.


## Installation

1. Ensure you have Poetry installed on your system.
2. Clone this repository to your local machine.
3. Navigate to the project directory in your terminal.
4. Run `poetry install` to install the dependencies.

## Running the Application

To run the application, use the following command:
```bash
poetry run streamlit run src/read_system/app.py
```

## Usage

1. Open the application in your web browser.
2. Paste your document content into the text area or upload a PDF file.
3. Ask a question about the content in the input field.
4. Click the "Run RAG" button to get an answer based on the document content.
5. The application will display the answer, top retrieved chunks, and chat history.

## Configuration

The application uses environment variables for configuration, particularly for the OpenAI API key. Ensure you have the necessary environment variables set before running the application.