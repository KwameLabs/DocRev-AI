import streamlit as st
from header import top_header

top_header()
st.header("Create User Account")
st.write(f"You are logged in as {st.session_state.role}.")
