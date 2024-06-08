import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

from typing import Any

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_model():
    model_names = {
    "gemini-1.0-pro":"Gemini 1.0 Pro",
    "gemini-1.5-flash-latest": "Gemini 1.5 Flash",
    "gemini-1.5-pro-latest":"Gemini 1.5 Pro",
    "gemini-pro-vision":"Gemini Vision Pro",    
    }
    selected_model_key: Any = st.sidebar.selectbox("Select the model to use:", list(model_names.values()))
    selected_model_id = [k for k, v in model_names.items() if v == selected_model_key][0]

    return selected_model_id


def get_conversational_chain(model_id):
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
    provided context, just say, "answer is not available in the context", don't provide the wrong answer.\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    #model = ChatGoogleGenerativeAI(model="gemini-1.0-pro", temperature=0.3)
    model = ChatGoogleGenerativeAI(model=model_id, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, model_id):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Allow dangerous deserialization
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain(model_id)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    st.write("### Reply:")
    st.write(response["output_text"])

def main():

    st.set_page_config(page_title="Chat with PDFs", page_icon="📄", layout="wide")
    st.title("Chat with PDF using Gemini Models 💁")

    st.sidebar.title("Menu:")
    model_id = get_model()
    pdf_docs = st.sidebar.file_uploader("Upload your PDF Files", accept_multiple_files=True)
    if st.sidebar.button("Process"):
        if pdf_docs:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Text extracted and processed successfully!")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question, model_id)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)


    st.markdown(
    """
    <div style="text-align:center">
        <p>Powered by <a href="https://deepmind.google/technologies/gemini/" target="_blank" style="text-decoration: none; color: white;">Gemini Models</a> And <a href="https://streamlit.io/" target="_blank" style="text-decoration: none; color: white;">Streamlit</a></p>
    </div>
    """,
    unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
