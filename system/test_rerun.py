import streamlit as st

# Initialize session state variables
if 'count_value' not in st.session_state:
    st.session_state.count_value = 0

# Callback functions
def increment_counter():
    st.session_state.count_value += 1
   
def decrement_counter():
    st.session_state.count_value -= 1

st.button('Increment', on_click=increment_counter, key='increment_btn')
st.button('Decrement', on_click=decrement_counter, key='decrement_btn')

# Print session state variable
st.write('Count = ', st.session_state.count_value)