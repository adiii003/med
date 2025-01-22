import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import time

# Replace with your actual API keys
GROQ_API_KEY = "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"
HUGGINGFACE_API_KEY = "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"

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(model_name="llama-3.1-8b-instant",api_key="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")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    If u feel that the question is relavant to the dataset availabe in the pdf dataset try helping the user and get teh job done
    <context>
    {context}
    <context>
    Question:{input}

    """
)

def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.loader = PyPDFDirectoryLoader("dataset")  # Replace "dataset" with the actual path to your dataset directory
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

create_vector_embedding()

st.set_page_config(page_title="MEDIC", page_icon="🏥", layout="wide")

st.markdown("""
    <style>
        .stTextInput>label {
            font-size: 24px; 
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
        .custom-header {
            font-size: 30px; 
            font-weight: bold;
            color: #333333;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="custom-header">Ask your medical queries here:</p>', unsafe_allow_html=True)

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'input' not in st.session_state:
    st.session_state.input = "" 

def process_input():
    user_input = st.session_state.input 
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': user_input}) 
        print(f"Response time :{time.process_time()-start}")

        st.session_state.messages.append({"role": "assistant", "content": response['answer']}) 
        st.session_state.input = "" 

for message in st.session_state.messages:
    col1, col2 = st.columns([1, 4]) 

    with col1:
        if message["role"] == "user":
            st.empty() 
        else:
            st.write("") 

    with col2:
        user_style = """
            background-color: #E8E8E8; 
            padding: 10px; 
            border-radius: 5px; 
            margin-bottom: 10px; 
            box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
        """
        bot_style = """
            background-color: #DCF8C6; 
            padding: 10px; 
            border-radius: 5px; 
            margin-bottom: 10px; 
            box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
        """
        if message["role"] == "user":
            st.markdown(
                f"<div style='{user_style}'>{message['content']}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='{bot_style}'>{message['content']}</div>",
                unsafe_allow_html=True,
            )

user_input = st.text_input(
    "", 
    value=st.session_state.input, 
    key="input", 
    on_change=process_input 
)

st.sidebar.title("MEDIC")
st.sidebar.info("Powered by Groq LLM")
