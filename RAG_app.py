import streamlit as st
import json
from langchain_openai import OpenAI,ChatOpenAI
from langchain_community.document_loaders import TextLoader, pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, FewShotPromptTemplate, PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_core.output_parsers.string import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback
from streamlit_lottie import st_lottie
import os
import time
import re
import string
from apikey import apikey


#OpenAI key
os.environ['OPENAI_API_KEY'] = apikey

llm=OpenAI()

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Function to preprocess text
# def preprocess_text(text):
#     # Step 1: Convert text to lowercase
#     text = text.lower()

#     # Step 2: Remove punctuation
#     text = text.translate(str.maketrans('', '', string.punctuation))

#     # Step 3: Remove extra whitespace and newlines
#     text = re.sub(r'\s+', ' ', text).strip()
    
lottie_file= os.path.abspath(r"C:\MyWorkspace\streamlit_apps\DocRev-AI\system\lottiefiles\loading.json")
lottie_loading = load_lottiefile(lottie_file)

st.title("DOCREV-AI App")
st_lottie(
    lottie_loading,
    speed=1,
    reverse=False,
    loop=True,
    quality="medium",
    height=100,
    width=100,
    key=None
)
#file = os.path.abspath(r"C:\MyWorkspace\streamlit_apps\DocRev-AI\system\pdfs\NHP_January_2020.pdf")

def load_documents(file):
    loader = pdf.PyMuPDFLoader(file)
    documents = loader.load()
    return documents
#processed_documents = preprocess_text(documents)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

docs = text_splitter.split_documents(load_documents())
model="text-embedding-3-small"
embedding = OpenAIEmbeddings(model=model)

vectorstore = FAISS.from_documents(docs, embedding)

memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True) 

query1 = "Does the policy document's Table of Content cover all necessary sections including Acknowledgements, Foreword/Preface, List of Tables and Figures, Acronyms, Executive Summary, Glossary, Introduction, Policy Context, Policy Framework, Strategies, Implementation Plan, Monitoring and Evaluation, and Communication Strategy? Does the policy document include all elements on the cover page as specified in the guidelines (Ghana Coat of Arms, institutional logo, name of institution, document title, effective date, and revised date)?Does the introduction chapter provide comprehensive background information, including a situational analysis, scope of the policy, process of preparing the policy, and content and structure? List all compliant and non-compliant sections under two sections with the headings Compliant Sections and Non-Compliant Sections in comma-separated format. Give advise on what needs to be done to improve the policy document. Place the answer in a table format with the respective headings."


#query_emb=FAISS.from_embeddings()


#query_answer = vectorstore.similarity_search(query1)



#docs_and_scores = vectorstore.similarity_search_with_score(query1)


#retriever = library.as_retriever()

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 2, "fetch_k": 2, "lambda_mult": 0.5},)

system_prompt = (
    "You are a public policy reviewer in the National Development Planning Commission of Ghana who needs to review a formulated public policy document."
    "Use the given context to answer the questions. "
    "If you don't know the answer, say unknown. "
    "If the context does not satisfy all the requirements don't say yes in the answer."
    "Context: {context}"
    "#Output layout"
    "Compliant"
    "   section"
    "Non-Compliant"
    "   Section"
    "What needs to be done"
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)



#retriever_query = "Does the policy document's Table of Content cover all necessary sections including Acknowledgements, Foreword/Preface, List of Tables and Figures, Acronyms, Executive Summary, Glossary, Introduction, Policy Context, Policy Framework, Strategies, Implementation Plan, Monitoring and Evaluation, and Communication Strategy? List all compliant and non-compliant sections under two sections with the headings Compliant and Non-Compliant."


question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
chain = create_retrieval_chain(retriever, question_answer_chain) 
#chain = prompt_template | model | StrOutputParser

with get_openai_callback() as cb:
    results = chain.invoke({"input": query1})


st.write(results['answer'])
print(cb)
