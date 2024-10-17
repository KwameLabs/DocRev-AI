import streamlit as st
import pandas as pd
import requests as rs

st.title('Amazing User Login App')

sheet_csv = st.secrets["database_url"]
res = rs.get(url=sheet_csv)
open('database.csv', 'wb').write(res.content)
database = pd.read_csv('database.csv', header=0)

# Create user_state
if 'user_state' not in st.session_state:
    st.session_state.user_state = {
        'name_surname': '',
        'password': '',
        'logged_in': False,
        'user_type': '',
        'mail_adress': '',
        'fixed_user_message': ''
    }

if not st.session_state.user_state['logged_in']:
    # Create login form
    st.write('Please login')
    mail_adress = st.text_input('E-Mail')
    password = st.text_input('Password', type='password')
    submit = st.button('Login')

    # Check if user is logged in
    if submit:
        user_ = database[database['mail_adress'] == mail_adress].copy()
        if len(user_) == 0:
            st.error('User not found')
        else:
            if user_['mail_adress'].values[0] == mail_adress and user_['password'].values[0] == password:
                st.session_state.user_state['mail_adress'] = mail_adress
                st.session_state.user_state['password'] = password
                st.session_state.user_state['logged_in'] = True
                st.session_state.user_state['user_type'] = user_['user_type'].values[0]
                st.session_state.user_state['mail_adress'] = user_['mail_adress'].values[0]
                st.session_state.user_state['fixed_user_message'] = user_['fixed_user_message'].values[0]
                st.write('You are logged in')
                st.rerun()
            else:
                st.write('Invalid username or password')
elif st.session_state.user_state['logged_in']:
    st.write('Welcome to the app')
    st.write('You are logged in as:', st.session_state.user_state['mail_adress'])
    st.write('You are a:', st.session_state.user_state['user_type'])
    st.write('Your fixed user message:', st.session_state.user_state['fixed_user_message'])
    if st.session_state.user_state['user_type'] == 'admin':
        st.write('You have admin rights. Here is the database')
        st.table(database)