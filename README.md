# Chat with PDF

Chat with PDF is a Streamlit application that allows users to upload a PDF, select the Gemini model, enter a query, and receive results. The app leverages the LangChain framework to handle various tasks related to text processing and interaction with the Gemini model.

## Features

- **Upload PDF**: Users can upload a PDF file for processing.
- **Select Gemini Model**: Choose the Gemini model to be used for generating responses.
- **Query Input**: Enter queries related to the content of the uploaded PDF.
- **Result Output**: Receive answers to queries based on the PDF content.

## Technologies Used

- **Streamlit**: For building and deploying the web application.
- **LangChain Framework**: For handling text processing and interactions with the Gemini model.
  - `langchain_text_splitters`: Used for splitting text into manageable chunks.
  - `GoogleGenerativeAIEmbeddings` and `ChatGoogleGenerativeAI` from `langchain_google_genai`: For embedding and chat functionalities.
  - `FAISS`: For efficient similarity search and clustering.
  - `load_qa_chain`: For loading the question-answering chain.
  - `PromptTemplate`: For creating prompt templates.

## Deployment

The application is deployed on Streamlit and can be accessed using the following URL:

[Chat with PDF - Streamlit App](https://chatwithpdff.streamlit.app/)

## How It Works

1. **Upload PDF**: Users start by uploading a PDF document they want to query.
2. **Model Selection**: Select the Gemini model for generating responses.
3. **Enter Query**: Input a query related to the content of the uploaded PDF.
4. **Receive Response**: The app processes the query and returns relevant answers based on the content of the PDF.

## Usage

1. Open the application using the provided URL.
2. Upload your PDF file.
3. Select the desired Gemini model.
4. Enter your query in the input field.
5. View the generated response.

## Acknowledgements

Special thanks to the developers of the LangChain framework and Streamlit for providing the tools necessary to build this application.

---
