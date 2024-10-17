
import streamlit as st
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import faiss
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PDFFileReader




# Text Loader - Load the text
def get_pdf_text(pdf_docs):
    text = ""
    page_count = 0
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
           # page_count += 1
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
    embeddings = OpenAIEmbeddings()
    vectorstore = faiss.FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore

#get the pdf texts
temp = open('data\Guidelines_for_Public_Policy_Formulation_in_Ghana_Final_Nov20201_ML.pdf', 'rb')
PDF_read = PDFFileReader(temp)
first_page = PDF_read.getPage(0)
print(first_page.extractText())

#pdf_docs=st.sidebar.file_uploader("Upload your Policy document PDF here and click on 'Review document'", accept_multiple_files=False,  key='pdf')
raw_text = get_pdf_text(temp)
