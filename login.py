import streamlit as st
import sqlite3


conn = sqlite3.connect('docrev.db', check_same_thread=False)
cursor = conn.cursor()

def userCreation():
    st.write('Please fill this form')
    with st.form(key='Registration form'):
     username = st.text_input('Enter your username')
     firstname = st.text_input('Enter your first name')
     lastname = st.text_input('Enter your last name')
     email = st.text_input('Enter your email')
     password = st.text_input('Enter your password', type='password')
     dateofregistration = st.date_input('Enter the date')
     #password = ''
     submit = st.form_submit_button(label='Register')
     
     if submit == True:
         st.success('Your registration has been successful')
         addInfo(username,firstname,lastname,email,password,dateofregistration)
    
    
def addInfo(a,b,c,d,e,f):
    cursor.execute(
        """
CREATE TABLE IF NOT EXISTS users(username TEXTS(20),firstname TEXTS(20),lastname TEXTS(20),email TEXTS(20),password TEXTS(20),dateofreg TEXTS(20))     
"""
                   )
    cursor.execute('INSERT INTO users VALUES(?,?,?,?,?,?)',(a,b,c,d,e,f))
    conn.commit()
    conn.close()
    #st.success('User has been registered')
    
def userLogin(username, password):
    cursor.execute('SELECT')

userCreation()