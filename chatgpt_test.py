import streamlit as st
import pickle as pk
import pandas as pd
import plotly.graph_objects as go
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

from langchain.text_splitter import CharacterTextSplitter
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
#import streamlit.components.v1 as components





#OpenAI key
os.environ['OPENAI_API_KEY'] = apikey

ROOT_DIR = os.path.abspath(os.curdir)
llm = openai.OpenAI(temperature=0)
pdf_docs = bytes()

# Declare variable.
if 'pdf_ref' not in ss:
    ss.pdf_ref = None

st.set_page_config(
    page_title="Public Policy Document Review System",
    page_icon="👀",
    layout="wide",
    initial_sidebar_state="collapsed",
    )


#logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Render images
#render_image("gh_coat.png")

#PDF viewer
container_pdf, container_chat = st.columns([50, 50])



# Text Loader - Load the text
def get_pdf_text(pdf_docs):
    text = ""
    page_count = 0
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
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
 

# Get vector store
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", show_progress_bar=True)
    #embeddings =  HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = faiss.FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vectorstore.save_local("faiss_index")
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
    
def process_question(question: str, vectorstore) ->str:
    """
    Process a user question using the vector database

    Args:
        question (str): The user's question
        vector_db (faiss): The vector database containing document embeddings

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
        Origninal question:{question}""",
    )
    
    
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever= vectorstore.as_retriever(),llm=llm, prompt=QUERY_PROMPT, parser_key="lines"
    )
    
    template = """Answer the question based ONLY on the following context :
    {context}
    Question: {question}
    Add snippets of the context you used to answer the question.
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
         ss.get('pdf_ref')[0].name }""")  
    
    pdf_pages = []
    with pdfplumber.open(pdf_docs) as pdf:
        pdf_pages = [page.to_image().original for page in pdf.pages]
    logging.info("PDF pages extracted  as images")
    return pdf_pages
    
    
    
def user_input(user_question):
    embeddings = OpenAIEmbeddings()
    
    new_db = faiss.FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    # new_db.s
    
    
    chain = get_conversation_chain()
    
    response = chain(
        {"input_documents":docs, "question": user_question},
        return_only_outputs=True   
    )
    print(response)
    st.write("Reply: ", response["output_text"])
    

    
def add_tabMenu():
    selected = option_menu(
    menu_title= None,
    options=["Policy Document", "Review Advisory"], 
    icons=["file-earmark-text","chat-square-text"],
    menu_icon="cast",
    default_index= 0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "14px"}, 
        "nav-link": {"font-size": "14px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#638FC2"},
    }
)
                   
    #data = get_clean_data()
    
def loaddoc(pdf_docs): 
    temp_dir = tempfile.mkdtemp() #temporary directory
    #path = os.path.join(temp_dir,ss.get('pdf_ref')[0].name)
    path = os.path.join(temp_dir,pdf_docs.name)
    with open(path, "wb") as f:
        f.write(pdf_docs.getvalue())
        logging.info(f"File saved to temporary path: {path}")
        # loader = UnstructuredPDFLoader(path)
        loader = PyPDFLoader(path)
        data = loader.load()
    return data   
    
    
    
    
      





def main() -> None:
    
    html = """
    <style>
        .reportview-container {
        flex-direction: row-reverse;
        }

        header > .toolbar {
        flex-direction: row-reverse;
        left: 1rem;
        right: auto;
        }

        .sidebar .sidebar-collapse-control,
        .sidebar.--collapsed .sidebar-collapse-control {
        left: auto;
        right: 0.5rem;
        }

        .sidebar .sidebar-content {
        transition: margin-right .3s, box-shadow .3s;
        }

        .sidebar.--collapsed .sidebar-content {
        margin-left: auto;
        margin-right: -21rem;
        }

        @media (max-width: 991.98px) {
        .sidebar .sidebar-content {
            margin-left: auto;
        }
        }
         div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
                    position: sticky;
                    top: 2.875rem;
                    background-color: white;
                    z-index: 999;
                }
                .fixed-header {
                    border-bottom: 1px solid black;
                }
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
                
                [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
         width: 450px;
       }
       [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
           width: 500px;
           margin-left: -500px;
        }
    </style>
    """
    st.markdown(html, unsafe_allow_html=True)
   
    st.markdown("<h1 style='text-align: center; color: black;'>👀 DocRev-AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: black;'>An AI-Assistant for Public Policy Document Review</p>", unsafe_allow_html=True)


    load_dotenv()
    
    
    
    
    st.subheader("", divider="gray", anchor=False,)
    st.write("")
    
    col1, col2 = st.columns([1.5,2])
    
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        
    if "vector_store" not in st.session_state:
        st.session_state["vector_store"] = None
        
    
    pdf_docs=col1.file_uploader("Upload your Policy document PDF here and click on 'Review document'", type="pdf", accept_multiple_files=False,  key='pdf')
    # if ss.pdf:
    #     ss.pdf_ref = ss.pdf  # backup
    
       

    
    if pdf_docs:
        
        
        raw_text = get_pdf_text(loaddoc(pdf_docs))
                
                #get the text chunks
        text_chunks = get_text_chunks(raw_text)
                #st.write(text_chunks)
                #create vector store
    
        vectorstore = get_vectorstore(text_chunks)
        st.session_state["file_upload"] = pdf_docs
        if st.session_state["vector_store"] is None:
            st.session_state["vector_store"]= vectorstore
        pdf_pages = extract_all_pages_as_images(pdf_docs)
        st.session_state["pdf_pages"] = pdf_pages
        
        zoom_level = col1.slider(
            "Zoom Level", min_value=100, max_value=1000, value=700, step=50
        )
        with col1:
         add_tabMenu()
         with st.container(height=410, border=True):
             for page_image in pdf_pages:
                 st.image(page_image, width=zoom_level)
        
        with col2:
           message_container = st.container(height=500, border=True)
           
           for message in st.session_state["messages"]:
               avatar ="🐙"  if message["role"] == "assistant" else "😄"
               with message_container.chat_message(message["role"], avatar=avatar):
                   st.markdown(message["content"])
                   
                   if prompt :=st.chat_input("Enter a prompt here..."):
                       try:
                           st.session_state["messages"].append(
                               {"role": "user", "content": prompt}
                           )
                           message_container.chat_message(
                               "user", avatar="😄").markdown(prompt)
                           
                           with message_container.chat_message("assistant", avatar="🐙"):
                               with st.spinner(":green[processing...]"):
                                   if st.session_state["vector_store"] is not None:
                                       response = process_question(
                                           prompt, st.session_state["vector_store"]
                                       )
                                       st.markdown(response)
                                   else:
                                       st.warning("Please upload a PDF file first.")
                           if st.session_state["vector_db"] is not None:
                              st.session_state["messages"].append(
                                {"role": "assistant", "content": response}
                              )
                       except Exception as e:
                            st.error(e, icon="⛔️")
                            logging.error(f"Error processing prompt: {e}")
                   else:
                        if st.session_state["vector_db"] is None:
                            st.warning("Upload a PDF file to begin chat...")
    
    if ss.pdf_ref:
        
       #binary_data = ss.get('pdf_ref')
       binary_data = ss.pdf_ref
       temp_dir = tempfile.mkdtemp() #temporary directory
       path = os.path.join(temp_dir,ss.get('pdf_ref')[0].name)
    #    bin_data = list(map(bin, binary_data))
       pdf_viewer(input=path, width=700, height=1000 )
    # pdf_viewer()
    #    pdf_viewer(os.path.join(ROOT_DIR, "system/pdfs/NHP_January_2020.pdf"), rendering='legacy_embed', height=500)
        # pdf_viewer()
        
    prompt = col2.text_input('Ask a question about the document')
    

    # llms
   
   # llm = openai.OpenAI(temperature=1.0)
    #title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True)
    # Show stuff to the screen if there's a prompt
    if prompt:
       # response = title_chain.run(topic=prompt)
       # st.write(response)
       user_input(prompt)
      # response=get_conversation_chain()
       #st.write(response)
    
    #components.iframe("https://ghana.gov.gh/", height=500)
    if st.sidebar.button("Review document"):
        if pdf_docs:
            with st.spinner("Processing..."):
                #get the pdf texts
                raw_text = get_pdf_text(pdf_docs)
                
                #get the text chunks
                text_chunks = get_text_chunks(raw_text)
                #st.write(text_chunks)
                #create vector store
                vectorstore = get_vectorstore(text_chunks)
                
                #file = open('.../app/faiss_index/index.pkl', 'rb')
                # dump information to that file
                #data = pk.load(file)
                with open('faiss_index\index.pkl', 'rb') as f:
                    pk_loaded = pk.load(f) # deserialize using load()
                    st.write(pk_loaded) # print student names
                # close the file
                f.close()
                

                
                #create conversation chain
                #conversation = get_conversation_chain(vectorstore)

                retriever = vectorstore.as_retriever()
        else:
            st.sidebar.warning("No document uploaded for review yet.")
   

#tests whether the file is being run directly or imported
if __name__ == '__main__':
    main()