import streamlit as st
import pickle as pk
import pandas as pd
#import plotly.graph_objects as go
import numpy as np
import os
import tempfile
import shutil
import logging
from apikey import apikey
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.llms import openai
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
import io
from dotenv import load_dotenv
from pypdf import PdfReader
import pdfplumber 
from langchain_community.document_loaders import TextLoader, pdf
from langchain_community.document_loaders import UnstructuredPDFLoader,OnlinePDFLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceInstructEmbeddings
from langchain.vectorstores import faiss
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains import RetrievalQA
from langchain.embeddings import sentence_transformer
from langchain.memory import ConversationBufferMemory
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.question_answering import load_qa_chain
from streamlit_option_menu import option_menu
from pathlib import Path
import streamlit_authenticator as stauth
from streamlit_pdf_viewer import pdf_viewer
from streamlit import session_state as ss
from image_loader import render_image
#import chromadb
from langchain_openai import ChatOpenAI 
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables  import RunnablePassthrough
from langchain_core.output_parsers.string import StrOutputParser
from typing import List, Tuple, Dict, Any, Optional
import charset_normalizer
from common import set_page_container_style
from langchain.callbacks import get_openai_callback
from langchain.chains.combine_documents import create_stuff_documents_chain
from docx_creator import create_word_doc
import threading
import time
import queue
from multiprocessing import Process
import faiss as fs


#import streamlit.components.v1 as components





#OpenAI key
os.environ['OPENAI_API_KEY'] = apikey

ROOT_DIR = os.path.abspath(os.curdir)
llm = openai.OpenAI(temperature=0)
pdf_docs = bytes()

# logo

#st.logo("images/DocRev-AI.png", icon_image="images/DocRev-AI_logo.png")



st.set_page_config(
    page_title="Public Policy Document Review System",
    page_icon="👀",
    layout="wide",
    initial_sidebar_state="expanded",
    )

set_page_container_style(
        max_width = 1100, max_width_100_percent = True,
        padding_top = 0, padding_right = 10, padding_left = 5, padding_bottom = 10
)

if 'review_advisory' not in st.session_state:
    st.session_state.review_advisory = None
#logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)



class FAISSVectorstore:
    def __init__(self, index, embeddings):
        self.index = index
        self.embeddings = embeddings

    def save_local(self, filepath):
        # Save the FAISS index to the specified location
        fs.write_index(self.index, filepath)
    
    def load_local(self, filepath):
        # Load the FAISS index from the specified location
        self.index = fs.read_index(filepath)
    
    def add_documents(self, text_chunks):
        # Convert document chunks to embeddings
        doc_embeddings = self.embeddings.embed_documents(text_chunks)
        doc_embeddings = np.array(doc_embeddings).astype("float32")
        
        # Add the document embeddings to the FAISS index
        self.index.add(doc_embeddings)

    def search(self, query_text, k=5):
        # Embed the query text
        query_embedding = self.embeddings.embed_query(query_text)
        query_embedding = np.array([query_embedding]).astype("float32")
        
        # Search the FAISS index for nearest neighbors
        distances, indices = self.index.search(query_embedding, k)
        return distances, indices

def about():
    st.sidebar.markdown('---')
    st.sidebar.info('''
    ### DocRev-AI App

    Updated: 9 October, 2024''')

# with st.sidebar:
#     selected = option_menu(
#         menu_title="Main Menu",
#         options=["Home","Logout"],
#         icons=["house-fill","door-closed-fill"],
#         menu_icon="search",
#         default_index=0
#     )



#PDF viewer
container_pdf, container_chat = st.columns([50, 50])

# def load_documents():
#     loader = pdf.PyMuPDFLoader(file)
#     documents = loader.load()
#     return documents

# Document loader - Load the document
def store_in_temp(pdf_docs):
    temp_dir = tempfile.mkdtemp() #temporary directory
    path = os.path.join(temp_dir,pdf_docs.name)
    return path

def store_in_temp2(pdf_docs):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(pdf_docs.getvalue())
        temp_file_path = temp_file.name
        # print(temp_file_path)
        return temp_file_path
    
def loaddoc(path): 
    #temp_dir = tempfile.mkdtemp() #temporary directory
    #path = os.path.join(temp_dir,ss.get('pdf_ref')[0].name)
    #path = os.path.join(temp_dir,pdf_docs.name)
    with open(path, "wb") as f:
        f.write(pdf_docs.getvalue())
        logging.info(f"File saved to temporary path: {path}")
        # loader = UnstructuredPDFLoader(path)
        #reader = PdfReader()
        loader = PyPDFLoader(path)
        data = loader.load()
    return data  

# Text Loader - Load the text
def get_pdf_text(path):
    text = ""
    page_count = 0
    #for pdf in path:
    pdf_reader = PdfReader(path)
    for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Text Splitter - Split the loaded text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap = 200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_document_chunks(document):
    document_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap = 200
    )
    chunks = document_splitter.split_documents(document)
    return chunks


# Get vector store
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", show_progress_bar=True)
    #embeddings =  HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = faiss.FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore

def get_vectorstore2(document_chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", show_progress_bar=True)
    #embeddings =  HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = faiss.FAISS.from_documents(documents=document_chunks, embedding=embeddings)
    
    vectorstore.save_local("faiss_index")
    return vectorstore


def get_vectorstore3(text_chunks):
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", show_progress_bar=True)
    
    # Convert document chunks to embeddings
    #doc_embeddings = embeddings.embed_documents(text_chunks)
    doc_embeddings = np.array(doc_embeddings).astype("float32")
    
    # Define the dimension of the embeddings
    d = len(doc_embeddings[0])
    
    # Number of clusters (nlist) for IVF
    nlist = 100  # Can be adjusted based on your data size
    
    # Create a FAISS index using IVF (IndexIVFFlat) for faster search
    quantizer = fs.IndexFlatL2(d)  # Quantizer used to assign vectors to clusters
    
    index = fs.IndexIVFFlat(quantizer, d, nlist, fs.METRIC_L2)
    
    # Train the index with the document embeddings (required for IVF)
    index.train(doc_embeddings)
    
    # Add the document embeddings to the index
    index.add(doc_embeddings)
    
    # Save the FAISS index locally
    fs.write_index(index, "faiss_index_ivfflat")

    # Create a FAISS vector store with the trained index and embeddings
    vectorstore = faiss.FAISS(embedding=embeddings, index=index)
    fs.ve
    
    return vectorstore
  

def get_vectorstore4(text_chunks):
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", show_progress_bar=True)
    
    # Convert document chunks to embeddings
    doc_embeddings = embeddings.embed_documents(text_chunks)
    doc_embeddings = np.array(doc_embeddings).astype("float32")
    
    # Define the dimension of the embeddings
    d = doc_embeddings.shape[1]  # Get dimensionality from embeddings shape
    
    # Number of clusters (nlist) for IVF
    nlist = 5  # Set this to be smaller than or equal to the number of documents
    
    # Create a FAISS index using IVF (IndexIVFFlat) for faster search
    quantizer = fs.IndexFlatL2(d)  # The quantizer assigns vectors to clusters
    index = fs.IndexIVFFlat(quantizer, d, nlist, fs.METRIC_L2)
    
    # Train the FAISS index on document embeddings
    index.train(doc_embeddings)
    
    # Add the document embeddings to the index
    index.add(doc_embeddings)
    
    # Save the FAISS index locally
    fs.write_index(index, "faiss_index_ivfflat")
    
    # Return a custom FAISS vector store object
    vectorstore = FAISSVectorstore(index=index, embeddings=embeddings)
    
    return vectorstore



    
def get_conversation_chain():
    prompt_template="""
    Answer the question as detailed as possible from the provided context,
    make sure to provide all the details, if the answer is not in the provided context
    just say, "answer is not available in the context", don't provide wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    
    Answer:
    
    """
    
    
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context","questions"])
    chain = load_qa_chain(llm,chain_type="stuff", prompt=prompt)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return chain
    #conversation_chain = conversational_retrieval.
    
def  process_question(question: str, vectorstore) ->str:
    """
    Process a user question using the vector database

    Args:
        question (str): The user's question
        vector_store (faiss): The vector database containing document embeddings

    Returns:
        str: The generated response to the user's question
    """
    
    logging.info(f"""Processing question: {
                question}""")
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI public policy review assistant. Your task is to generate different versions 
        of the given user question to retrieve relevant documents from a vector database. By generating multiple
        perspectives on the user question, your goal is to help the user overcome some of the limitations of distance-based
        similarity search. Provide these alternatives questions separated by newlines.
        Original question:{question}""",
    )
    
    
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever= vectorstore.as_retriever(),llm=llm, prompt=QUERY_PROMPT, parser_key="lines",
    )
    
    template = """Answer the question based ONLY on the following context. If the answer begins with a yes, don't include it. :
    {context}
    Question: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {"context": retriever_from_llm, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    response = chain.invoke(question)
    logging.info("Question processed and response generated")
    return response

def process_question3(question: str, vectorstore) -> str:
    """
    Process a user question using the FAISS vector database.

    Args:
        question (str): The user's question.
        vectorstore: The FAISS vector database containing document embeddings.

    Returns:
        str: The generated response to the user's question.
    """
    
    logging.info(f"Processing question: {question}")
    
    # Generate alternative questions to retrieve relevant documents
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI public policy review assistant. Your task is to generate different versions 
        of the given user question to retrieve relevant documents from a vector database. By generating multiple
        perspectives on the user question, your goal is to help the user overcome some of the limitations of distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}"""
    )
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", show_progress_bar=True)
    # Assume that embeddings need to be generated externally using the same model used for indexing
    logging.info("Generating question embeddings using the external embeddings model...")
    embeddings_model = vectorstore  # Assuming you have stored the embeddings model separately
    #embeddings_model = embeddings
    
    question_embedding = embeddings_model.embed_query(question)
    
    # Perform a search on the FAISS index with the question embedding
    logging.info("Searching the FAISS index for relevant documents...")
    distances, indices = vectorstore.index.search(np.array([question_embedding]), k=5)

    # Retrieve the corresponding documents based on the indices
    relevant_documents = [vectorstore.documents[i] for i in indices[0] if i != -1]
    
    # Prepare the context from the retrieved documents
    context = "\n".join([str(doc) for doc in relevant_documents])

    # Define the prompt template for generating the final response
    template = """Answer the question based ONLY on the following context. If the answer begins with a yes, don't include it:
    {context}
    Question: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create the chain for generating the response
    chain = (
        {"context": context, "question": question}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    response = chain.invoke(question)
    
    logging.info("Question processed and response generated.")
    
    return response


def process_question2(question: str, vectorstore) -> str:
    """
    Process a user question using the vector database.

    Args:
        question (str): The user's question.
        vector_store (FAISSVectorstore): The vector database containing document embeddings.

    Returns:
        str: The generated response to the user's question.
    """
    
    logging.info(f"Processing question: {question}")
    
    # Generate alternative questions to retrieve relevant documents
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI public policy review assistant. Your task is to generate different versions 
        of the given user question to retrieve relevant documents from a vector database. By generating multiple
        perspectives on the user question, your goal is to help the user overcome some of the limitations of distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}"""
    )
    
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(), llm=llm, prompt=QUERY_PROMPT, parser_key="lines",
    )

    # Define the prompt template for generating the final response
    template = """Answer the question based ONLY on the following context. If the answer begins with a yes, don't include it:
    {context}
    Question: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Using the retriever from FAISS to search for documents relevant to the query
    logging.info("Retrieving documents from the vectorstore...")
    relevant_context, _ = vectorstore.search(question, k=5)
    
    # Process retrieved documents into context for answering the question
    context = "\n".join([str(doc) for doc in relevant_context])  # Assumes documents are in readable string format
    
    # Create the chain for generating the response
    chain = (
        {"context": context, "question": question}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    response = chain.invoke(question)
    
    logging.info("Question processed and response generated.")
    
    return response




@st.cache_data
def extract_all_pages_as_images(pdf_docs) -> List[Any]: 
    """
    Extract all pages from a PDF file as images.

    Args:
        pdf_docs (st.UploadFile): Streamlit file upload object containing 

    Returns:
        List[Any]: A list of image objects representing each page of the PDF.
    """
    logging.info(f"""Extracting all pages as images from file: {
         pdf_docs.name }""")  
    
    pdf_pages = []
    with pdfplumber.open(pdf_docs) as pdf:
        pdf_pages = [page.to_image().original for page in pdf.pages]
    logging.info("PDF pages extracted  as images")
    return pdf_pages
    
    
    
def user_input(user_question):
    embeddings = OpenAIEmbeddings()
    
    new_db = faiss.FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
  
    
    
    chain = get_conversation_chain()
    
    response = chain(
        {"input_documents":docs, "question": user_question},
        return_only_outputs=True   
    )
    print(response)
    st.write("Reply: ", response["output_text"])
    
  
                    
    #data = get_clean_data()
    
def generateReviewAdvisory(file) -> str:
   
    loader = pdf.PyMuPDFLoader(file)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    )
    
    docs = text_splitter.split_documents(documents)
    model="text-embedding-3-small"
    embedding = OpenAIEmbeddings(model=model)
    
    vectorstore = faiss.FAISS.from_documents(docs, embedding)
    

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True) 

    query1 = "Does the policy document's Table of Content cover all necessary sections including Acknowledgements, Foreword/Preface, List of Tables and Figures, Acronyms, Executive Summary, Glossary, Introduction, Policy Context, Policy Framework, Strategies, Implementation Plan, Monitoring and Evaluation, and Communication Strategy? Does the policy document include all elements on the cover page as specified in the guidelines (Ghana Coat of Arms, institutional logo, name of institution, document title, effective date, and revised date)?Does the introduction chapter provide comprehensive background information, including a situational analysis, scope of the policy, process of preparing the policy, and content and structure? List all compliant and non-compliant sections under two sections with the headings Compliant Sections and Non-Compliant Sections in comma-separated format. Give advise on what needs to be done to improve the policy document. Place the answer in a table format with the respective headings."
    
    retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 2, "fetch_k": 2, "lambda_mult": 0.5},)
    
    system_prompt = (
    "You are a public policy reviewer in the National Development Planning Commission of Ghana who needs to review a formulated public policy document."
    "Use the given context to answer the questions. "
    "If you don't know the answer, say unknown. "
    "If the context does not satisfy all the requirements don't say yes in the answer. If satisfies, don't include yes in your answer."
    "Context: {context}"
    "#Output layout"
    "Compliant"
    "   section"
    "Non-Compliant"
    "   Section"
    "Recommendation"
    )
    
    prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
    chain = create_retrieval_chain(retriever, question_answer_chain) 
    
    # chain = (
    #     {"context": retriever, "question": RunnablePassthrough()}
    #     | prompt_template
    #     | llm
    #     | StrOutputParser()
    # )
    

    with get_openai_callback() as cb:
        results = chain.invoke({"input": query1})
        
    return results



# t1 = threading.Thread(target=generateReviewAdvisory)
# t2 = threading.Thread(target=loaddoc)
def main() -> None:
    
    
    
    menu = ["Home", "Logout"]
    #about()
    color = '#262730'
    html = """
    <style>
             
        .header {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        background-color: #99ff99;
        padding: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: #011936;
        z-index: 1000;
        box-shadow: 0 4px 2px -2px gray;
        
    }
    
      .header {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        background-color: #1C3144;
        padding: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: #fff;
        z-index: 1000;
        box-shadow: 0 4px 2px -2px gray;
    }
    
    #MainMenu {visibility:hidden;}
    footer {visibility:hidden;}
    header {visibility:hidden;}
    
    /* Add some padding below the header to prevent content overlap */
    .main-content {
        margin-top: 70px;
    }
    
   .reportview-container .css-1lcbmhc .css-1outpf7 {{
                    padding-top: 35px;
                }}
                .reportview-container .main .block-container {{
                    {max_width_str}
                    padding-top: {padding_top}rem;
                    padding-right: {padding_right}rem;
                    padding-left: {padding_left}rem;
                    padding-bottom: {padding_bottom}rem;
                }}
                .reportview-container .main {{
                    color: {color};
                    background-color: {background_color};
                }}
    
    .block-container {
                    padding-top: 4rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                    
                }
                
    .rightsidebar {
        padding-top: 4rem;
        position: fixed;
        width: 200px;
        height: 400px;
        background: #000;
        margin-left: auto; 
        margin-right: 0;
    }
    
    [data-testid="stSidebarNav"] {{
    position:absolute;
    bottom: 0;
    z-index: 1;
    background: {color};
    }}
    [data-testid="stSidebarNav"] > ul {{
        padding-top: 2rem;
    }}
    [data-testid="stSidebarNav"] > div {{
        position:absolute;
        top: 0;
    }}
    [data-testid="stSidebarNav"] > div > svg {{
        transform: rotate(180deg) !important;
    }}
    [data-testid="stSidebarNav"] + div {{
        overflow: scroll;
        max-height: 66vh;
    }}
    
    div:has( >.element-container div.floating) {
    display: flex;
    flex-direction: column;
    position: fixed;
    }

    div.floating {
        height:0%;
    }
    </style>
    
    <div class="header">
        DocRev-AI: AI-Assisted Public Policy Review System for Ghana
    </div>
    
    """
    st.markdown(html, unsafe_allow_html=True)
    
    # Render images
    #render_image("images/DocRev-AI_header2.png")
    #render_image("images/horizontal_blue_1.png")
    
    #st.markdown("<h1 style='text-align: center; color: black;'>👀 DocRev-AI</h1>", unsafe_allow_html=True)
    #st.markdown("<h6 style='text-align: left; color: black;'>AI-Assisted Compliance Review for National Public Policy Formulation Standards in Ghana</h6>", unsafe_allow_html=True)


    load_dotenv()
    
   # Sidebar for navigation and zoom
    
st.sidebar.title("Reviewer Menu")

if 'page' not in st.session_state:
    st.session_state.page = ""
    
# Callback functions
def change_page(page_value):
    st.session_state.page = ''
#page = st.sidebar.radio("Go to", ("Document Upload", "Chat with Assistant", "Final Review Advisory")) 
st.session_state.page = st.sidebar.radio("Go to", ("Document Upload & Chat", "Final Review Advisory")) 
    
#render_image("images/DocRev-AI_header2.png")
#st.text("AI-Assisted Compliance Review for National Public Policy Formulation Standards in Ghana")    
#st.subheader("", divider="gray", anchor=False,)

    
    # Sticky header
st.markdown("<div class='main-content'>", unsafe_allow_html=True)

# Section for system purpose and introduction
#st.markdown("### Welcome to the AI-Assisted Public Policy Compliance Review System")
st.markdown("""
    This system helps public policy reviewers assess compliance with national standards in public policy formulation.
    Use the sections below to upload, review, and chat with the assistant to ensure all standards are met.
""")

with st.container(height=270, border=True):
        st.subheader("Upload the document you want to assess")
        
        pdf_docs=st.file_uploader("Upload Policy document for review here!'", type=['pdf', 'docx'], accept_multiple_files=False,  key='pdf')
col1, col2 = st.columns([2.5, 1.5])

if "messages" not in st.session_state:
        st.session_state["messages"] = []

if "vector_store" not in st.session_state:
        st.session_state["vector_store"] = None
        
if "file_name" not in st.session_state:
        st.session_state["file_name"] = None
        
if "zoom_level" not in st.session_state:
        st.session_state["zoom_level"] = 600
        
# Section for document upload
if st.session_state.page == "Document Upload & Chat":
   
   # Display document content (dummy text here for placeholder)
    with col1:
        st.markdown("""
        <style>
        .big-font {
            font-size:22px !important;
            font-weight: bold !important;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<p class="big-font">Document Preview</p>', unsafe_allow_html=True)
    #col1.subheader('Document Preview')
    image_container = col1.container(height=510, border=True)
    
    if pdf_docs is not None:
        
        path = store_in_temp2(pdf_docs)
                    
        if st.session_state["file_name"] is None:
            st.session_state["file_name"] = pdf_docs.name
            result = generateReviewAdvisory(path)
            # p = Process(target=generateReviewAdvisory, args=('path',))
            # p.start()
            # p.join()
            
        if st.session_state["review_advisory"] is None:
            st.session_state["review_advisory"] = result['answer']
        
        
        #st.success("File uploaded successfully!")
        #get the document
        document = loaddoc(path)
        text = get_pdf_text(path)
                
        #get the text chunks
        #document_chunks = get_document_chunks(document)
        text_chunks = get_text_chunks(text)
    
        
        vectorstore = get_vectorstore4(text_chunks)
        
        # Save the vector store
        vectorstore.save_local("faiss_index_ivfflat")
        
        
        
        st.session_state["file_upload"] = pdf_docs
        if st.session_state["vector_store"] is None:
            st.session_state["vector_store"]= vectorstore
        pdf_pages = extract_all_pages_as_images(pdf_docs)
        st.session_state["pdf_pages"] = pdf_pages
        
        if st.session_state["zoom_level"] is not None:
            st.session_state["zoom_level"] = st.sidebar.slider(
            "Zoom Level", min_value=100, max_value=1000, value=600, step=50
        )
        
        
        
        
         #add_tabMenu()
        with col1:
            with image_container:
                for page_image in pdf_pages:
                 st.image(page_image, width=st.session_state["zoom_level"])
    else:
        st.session_state["review_advisory"] = None

    # Section for displaying the document content
#elif page == "Chat with Assistant":
    with col2:
        st.markdown("""
        <style>
        .big-font {
            font-size:22px !important;
            font-weight: bold !important;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<p class="big-font">Chatting Area</p>', unsafe_allow_html=True)
    #col2.subheader("Chatting Area")
    
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        
    if "vector_store" not in st.session_state:
        st.session_state["vector_store"] = None
        
    with col2:  
    
        message_container = col2.container(height=450, border=True)
           
        for message in st.session_state["messages"]:
            avatar ="🤖"  if message["role"] == "assistant" else "👨"
            with message_container.chat_message(message["role"], avatar=avatar):
                 col2.markdown(message["content"])
                   
        if prompt :=col2.chat_input("Enter a prompt here to chat with AI..."):
            try:
                 st.session_state["messages"].append({"role": "user", "content": prompt})
                 message_container.chat_message("user", avatar="👨").markdown(prompt)
                           
                 with message_container.chat_message("assistant", avatar="🤖"):
                     with st.spinner(":green[processing...]"):
                        if st.session_state["vector_store"] is not None:
                            response = process_question3(
                              prompt, st.session_state["vector_store"]
                            )
                            st.markdown(response)
                        else:
                            st.warning("Please upload a PDF file first.")
                            
                 if st.session_state["vector_store"] is not None:
                     st.session_state["messages"].append(
                        {"role": "assistant", "content": response}
                     )
            except Exception as e:
                col2.error(e, icon="⛔️")
                logging.error(f"Error processing prompt: {e}")
        else:
             if st.session_state["vector_store"] is None:
                 col2.warning("Upload a PDF file to begin chat...")
  
    
# Section for final review advisory
elif st.session_state.page == "Final Review Advisory":
    st.subheader("Final Review Advisory")
    
    
    # Submit final review button
    # if st.button("Submit Final Review"):
    #     st.success("Your review has been submitted successfully!")
    st.write(st.session_state["review_advisory"])
    st.markdown("--------")
    st.write("""
        This section will summarize your findings and give you an advisory on the policy’s compliance.
        You can finalize your review based on the assistant’s suggestions and your own assessment.
    """)
    
    st.text_area("Write your final comment here", placeholder="Summarize your findings and advisory...")
    if st.session_state["review_advisory"] is not None:
     if st.button("Download Review Advisory (word format)"):
        create_word_doc(st.session_state["review_advisory"],st.session_state["file_name"])
        st.success('File downloaded successfully.')

# Closing main-content div
    st.markdown("</div>", unsafe_allow_html=True)

    
   
                


#tests whether the file is being run directly or imported
if __name__ == '__main__':
    #st.sidebar.image('./images/logo.png', output_format='png')
    
    main()
    
