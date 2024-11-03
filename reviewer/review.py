import streamlit as st
from header import *
import pickle as pk
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os
import tempfile
import shutil
import logging

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
#from streamlit_option_menu import option_menu
from pathlib import Path
#import streamlit_authenticator as stauth
from streamlit_pdf_viewer import pdf_viewer
from streamlit import session_state as ss
#from image_loader import render_image
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
from langchain_core.messages import HumanMessage, AIMessage
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


#########Initialiasation#################
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
ROOT_DIR = os.path.abspath(os.curdir)
#llm = openai.OpenAI(temperature=0)
llm = ChatOpenAI(model="gpt-4o-mini",temperature=1)
pdf_docs = bytes()

#########################################

     
# set_page_container_style(
#         max_width = 1100, max_width_100_percent = True,
#         padding_top = 0, padding_right = 10, padding_left = 5, padding_bottom = 10
# )

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
# )

load_dotenv()


# # Document loader - Load the document
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

# # Text Loader - Load the text
def get_pdf_text(path):
    text = ""
    page_count = 0
    #for pdf in path:
    pdf_reader = PdfReader(path)
    for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_document_chunks(document):
    document_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap = 200
    )
    chunks = document_splitter.split_documents(document)
    return chunks

# Returns vectorstore
# def get_vectorstore(document_chunks):
#     embeddings = OpenAIEmbeddings(model="text-embedding-3-small", show_progress_bar=True)
#     #embeddings =  HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
#     vectorstore = faiss.FAISS.from_documents(documents=document_chunks, embedding=embeddings)
    
#     vectorstore.save_local("faiss_index")
#     return vectorstore

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Returns retriever
def get_vectorstore2(document_chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", show_progress_bar=True)
    #embeddings =  HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = faiss.FAISS.from_documents(documents=document_chunks, embedding=embeddings)
    
    vectorstore.save_local("faiss_index")
    # retriever = vectorstore.as_retriever()
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 8, "lambda_mult": 0.7}
    )
    return retriever

# get response
def process_question2(question: str, retriever) ->str:
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # """
    # Process a user question using the vector database

    # Args:
    #     question (str): The user's question
    #     vector_store (faiss): The vector database containing document embeddings

    # Returns:
    #     str: The generated response to the user's question
    # """
    
    logging.info(f"""Processing question: {
                question}""")

    # template = """
    # Use the following pieces of context to answer the question at the end.
    # If you don't know the answer, just say that you do not have the relevant information needed to provide a verified answer, don't try to make up an answer.
    # When providing an answer, aim for clarity and precision. Position yourself as a knowledgeable authority on the topic, but also be mindful to explain the information in a manner that is accessible and comprehensible to those without a technical background.
    # Always say "Do you have any more questions pertaining to this instrument?" at the end of the answer.
    # {context}
    # Question: {question}
    # Helpful Answer:"""
    
    template = """
     You are an AI public policy review assistant. Answer the question based ONLY on the following context:{context} 
     Question: {question}
     """
    
    rag_custom_prompt = PromptTemplate.from_template(template)
    
    # Set up the RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()} | 
        rag_custom_prompt | 
        llm
    )
    
    # Invoke the RAG chain with the question
    answer = rag_chain.stream(question)
    return answer


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
    
    
#### Generate Final Review Advisory ############
def generateReviewAdvisory(file) -> str:
    # Load and process document
    loader = pdf.PyMuPDFLoader(file)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    docs = text_splitter.split_documents(documents)
    
    # Embed documents
    model = "text-embedding-3-small"
    embedding = OpenAIEmbeddings(model=model)
    vectorstore = faiss.FAISS.from_documents(docs, embedding)
    
    # Define query
    query1 = (
        "Does the policy document's Table of Content cover all necessary sections including Acknowledgements, Foreword/Preface, List of Tables and Figures, Acronyms, Executive Summary, Glossary, Introduction, Policy Context, Policy Framework, Strategies, Implementation Plan, Monitoring and Evaluation, and Communication Strategy?"
        "Does the policy document include all elements on the cover page as specified in the guidelines (Ghana Coat of Arms, institutional logo, name of institution, document title, effective date, and revised date)? Does the introduction chapter provide comprehensive background information, including a situational analysis, scope of the policy, process of preparing the policy, and content and structure?"
        "Does the introduction chapter provide comprehensive background information, including a situational analysis, scope of the policy, process of preparing the policy, and content and structure?"
        "Is there a detailed discussion on the legal basis for the policy and relevant national, regional, and global frameworks that affect it?"
        "Does the policy framework chapter clearly outline the vision, goal, key objectives, core values, and guiding principles of the policy?"
        "Are the strategies to achieve the key policy objectives clearly defined and detailed in the document?"
        "Does the implementation plan provide clear implementation directions, step-by-step actions, and specify institutional arrangements and resource mobilization strategies?"
        "Is there a comprehensive plan for monitoring and evaluation, including a schedule for periodic reviews?"
        "Does the communication strategy effectively outline methods for disseminating the policy, creating awareness, and generating stakeholder feedback?"
    )
    
    # Embed query and retrieve documents
    query_embeddings = embedding.embed_query(query1)  # Confirm query1 is a string here
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 8, "lambda_mult": 0.7}
    )
    matched_docs = retriever.get_relevant_documents(query=query1)
    
    # Define the template for evaluating policy document compliance
    template = """
    Review the policy document and assess whether it complies with the following guidelines for public policy formulation. For each section, indicate **(passed)** if the document meets the guideline and **(not passed)** if it does not.
    "Use the the context provided to answer the questions. Do not be too strict to match the exact words in the question. If for example the question ask for revision date but the context has revised date, label it as passed."
    "Context: {context} "
    1. **Cover Page Elements:**
    - Ghana Coat of Arms
    - Institutional logo
    - Name of institution
    - Document title
    - Effective date (MM/YYYY)
    - Revised date (MM/YYYY)

    **Result:** 

    2. **Table of Contents:**
    - Acknowledgements
    - Foreword/Preface
    - List of Tables and Figures
    - Acronyms
    - Executive Summary
    - Glossary
    - Introduction
    - Policy Context
    - Policy Framework
    - Strategies
    - Implementation Plan
    - Monitoring and Evaluation
    - Communication Strategy

    **Result:** 

    3. **Introduction Chapter:**
    - Background information (includes rationale and relevant data)
    - Situational analysis (includes SWOT analysis)
    - Scope of the policy
    - Process of preparing the policy
    - Content and structure of the document

    **Result:** 

    4. **Policy Context (Chapter Two):**
    - Discussion of the legal basis
    - Mention of relevant national, regional, or global frameworks
    - Inclusion of information from treaties or conventions Ghana has ratified

    **Result:** 

    5. **Policy Framework (Chapter Three):**
    - Vision
    - Goal
    - Key Objectives
    - Core Values and Guiding Principles

    **Result:** 

    6. **Strategies to Achieve Objectives (Chapter Four):**
    - Clear documentation of strategies for each policy objective

    **Result:** 

    7. **Implementation Framework/Plan (Chapter Five):**
    - Clear steps for policy implementation
    - Institutional arrangements with defined roles
    - Resource mobilization plan

    **Result:** 

    8. **Monitoring and Evaluation (Chapter Six):**
    - Monitoring and evaluation plan
    - Policy review timeline

    **Result:** 
    9. **Communication Strategy (Chapter Seven):**
    - Strategy for communicating policy goals and objectives to stakeholders
    - Awareness creation plan for stakeholders’ roles
    - Feedback mechanism

    **Result:** 

    ---

    ### **Recommendation Section:**

    Based on the assessment, the following areas need improvement to ensure full compliance with the standards:

    1. **Areas Not Passed:** 
    2. **Suggestions for Improvement:** 
    """
    
    
    
    # prompt_template = ChatPromptTemplate.from_messages([
    #     ("system", system_prompt),
    #     ("human", "{input}"),
    # ])
    
    prompt_template = PromptTemplate.from_template(template=template)

    # Run the chain with simplified direct input
    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
    results = question_answer_chain.stream({"input": query1, "context": matched_docs})

    return results


st.subheader(f"Welcome, {st.session_state.user}.")

if st.session_state.role == "Admin":
    st.write(f"You are logged in as an {st.session_state.role}.")
elif st.session_state.role =="Reviewer":
    st.write(f"You are logged in as a {st.session_state.role}.")
    
st.markdown("""
    This system helps public policy reviewers in Ghana to assess compliance with national standards in public policy formulation.
    Use the sections below to upload, review, and chat with the assistant to ensure all standards are met.
""")

if 'page' not in st.session_state:
    st.session_state.page = ""
    
# # Callback functions
st.session_state.page = st.sidebar.radio("Go to", ("Document Upload & Chat", "Final Review Advisory"))

with st.container(height=270, border=True):

    st.markdown("""
                <style>
                .big-font {
                    font-size:18px !important;
                    font-weight: bold !important;
                }
                </style>
                """, unsafe_allow_html=True)

    st.markdown('<p class="big-font">Upload the document you want to assess</p>', unsafe_allow_html=True)
    
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
        
# # Section for document upload
if st.session_state.page == "Document Upload & Chat":
   

#    # Display document content (dummy text here for placeholder)

    with col1:
        st.markdown("""
        <style>
        .big-font {
            font-size:18px !important;
            font-weight: bold !important;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<p class="big-font">Document Preview</p>', unsafe_allow_html=True)

    image_container = col1.container(height=510, border=True)
    
    if pdf_docs is not None:
        
        path = store_in_temp2(pdf_docs)
                    
        if st.session_state["file_name"] is None:
            st.session_state["file_name"] = pdf_docs.name
            
        with col1:
            st.success("File uploaded successfully!", icon="✅")
#         #get the document
        document = loaddoc(path)
        text = get_pdf_text(path)
                
#         #get the text chunks
        document_chunks = get_document_chunks(document)
#         #text_chunks = get_text_chunks(text)
    
        
        #vectorstore = get_vectorstore(document_chunks)
        retriever = get_vectorstore2(document_chunks)
        
#         # Save the vector store
#         #vectorstore.save_local("faiss_index_ivfflat")
        
        
        
        st.session_state["file_upload"] = pdf_docs
        if st.session_state["vector_store"] is None:
            st.session_state["vector_store"]= retriever
        
        
        
        pdf_pages = extract_all_pages_as_images(pdf_docs)
        st.session_state["pdf_pages"] = pdf_pages
                       
        
        if st.session_state["zoom_level"] is not None:
            st.session_state["zoom_level"] = st.sidebar.slider(
            "Zoom Level", min_value=100, max_value=1000, value=600, step=50
        )
        
        with col1:
            with image_container:
                with st.spinner(":green[generating document image...]"):
                    for page_image in pdf_pages:
                        st.image(page_image, width=st.session_state["zoom_level"])
    else:
        st.session_state["review_advisory"] = None

  
   
        
#     if "vector_store" not in st.session_state:
#         st.session_state["vector_store"] = None
        
    with col2:  
        st.markdown("""
        <style>
        .big-font {
            font-size:18px !important;
            font-weight: bold !important;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<p class="big-font">Chatting Area</p>', unsafe_allow_html=True)
        
        message_container = col2.container(height=450, border=True)
           
                   
        if user_query :=col2.chat_input("Chat with DocRev-AI..."):
            try:
                # conversation
                with message_container:
                    for message in st.session_state.chat_history:
                        if isinstance(message, HumanMessage):
                            with st.chat_message("Human"):
                                st.markdown(message.content)
                        else:
                            with st.chat_message("AI"):
                                st.markdown(message.content)

                    # user input    
                    if user_query is not None and user_query !="":
                        st.session_state.chat_history.append(HumanMessage(user_query))

                        with st.chat_message("Human"):
                            st.markdown(user_query)
                        
                        with st.chat_message("AI"):
                             with st.spinner(":green[replying...]"):
                                #ai_response = st.write_stream(get_response2(user_query, st.session_state.chat_history, st.session_state["vector_store"]))
                                ai_response = st.write_stream(process_question2(user_query, st.session_state["vector_store"]))
                            # for ai_response in get_response3(
                            #                 user_question=user_query, 
                            #                 chat_history=st.session_state.chat_history, 
                            #                 vectorstore=st.session_state["vector_store"]
                            #             ):
                                st.session_state.chat_history.append(AIMessage(ai_response)) 
            except Exception as e:
                col2.error("Error processing prompt. You need to upload a document to chat.", icon="⛔️")
                logging.error(f"Error processing prompt: {e}")
        else:
             if st.session_state["vector_store"] is None:
                 col2.warning("Upload a PDF file to begin chat...", icon="⚠️")
  
    
# # Section for final review advisory
elif st.session_state.page == "Final Review Advisory":
   
    
    
#     # Submit final review button
#     # if st.button("Submit Final Review"):
#     #     st.success("Your review has been submitted successfully!")
#Generate review advisory 
    if "review_advisory" not in st.session_state:
        st.session_state["review_advisory"] = None
        
    if pdf_docs:
        st.subheader("Final Review Advisory")
        # if st.session_state["review_advisory"] is None:
        path = store_in_temp2(pdf_docs)
        result = generateReviewAdvisory(path)
        st.session_state["review_advisory"] = result
        with st.container(height=400):
            st.write_stream(st.session_state["review_advisory"])
                     
            # st.write(st.session_state["review_advisory"])
            #st.rerun()
       
       
        # with st.form("Reviewer_Remarks"):
        #             reviewer_remarks = st.text_area("Write your final comment/remarks here", placeholder="Summarize your findings and advisory...")
        #             add_remark = st.form_submit_button("Save Remarks")
        #             if add_remark:
        #                 final_remarks = reviewer_remarks
        with st.form(key='my_form'):                
                submitted = st.form_submit_button(label="Download Review Advisory (word format)")
                if submitted:
                    try:
                        create_word_doc(st.session_state["review_advisory"],st.session_state["file_name"])
                        st.success('File downloaded successfully.', icon="✅")  
                    except Exception as e:
                            st.error("Download folder path not accessible.", icon="⛔️")
           
        # Example of how to use the prompt with values

#     st.markdown("--------")
        # st.subheader("Reviewer's Remarks")
        # st.write("""
        #     Please use this section to give your final reviewer's remarks based on your findings.
        #     You can finalize your review based on the assistant’s suggestions and your own assessment.
        # """)
        
    else:
        st.warning("You must upload a public policy document to assess.", icon="⚠️")
        
    
    
            #     reviewer_remarks = st.text_area("Write your final comment/remarks here", placeholder="Summarize your findings and advisory...")
            #     add_remark = st.form_submit_button("Save Remarks")
            #     if add_remark:
            #         final_remarks = reviewer_remarks
                   
        

# # Closing main-content div
    st.markdown("\n\n", unsafe_allow_html=True)



