import streamlit as st
from header import top_header


top_header()
st.header("User Guide")

st.write('DocRev-AI is an AI-Assisted Public Policy Review System for Ghana. This guide takes you through the basics of the system and helps you to get moving with your review.')


st.subheader('Policy Document Upload')
st.write('Follow the steps below to upload the formulated policy document you want to assess.')
st.write('Step 1: Click on the "browse files" button.')
st.write('Step 2: From the open dialog box, double-click on the document or Select the document and click on the "open" button')
st.image('images/document_upload.png', caption='Document upload')
