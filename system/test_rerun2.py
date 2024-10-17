import streamlit as st

# Initialize session state variables
if 'count' not in st.session_state:
   st.session_state.count = 0


# Update session state variables
if st.button('Increment'):
   st.session_state.count += 1

if st.button('Decrement'):
   st.session_state.count -= 1

# Print session state variable
st.write('Count = ', st.session_state.count)