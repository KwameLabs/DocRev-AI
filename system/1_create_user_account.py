import streamlit as st
from pypdf import PdfReader
#from db_functions import *
import datetime
import hashlib

import sqlite3

try:
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
except Exception as e:
  print(e)
    



#st.set_page_config( initial_sidebar_state="collapsed")
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://i.postimg.cc/4xgNnkfX/Untitled-design.png");
background-size: cover;
background-position: center center;
background-repeat: no-repeat;
background-attachment: local;
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
</style>
"""

def create_user():
    c.execute("CREATE TABLE IF NOT EXISTS users(name TEXT, username TEXT, password TEXT,email TEXT,usertype TEXT, dateofreg TEXT)")
    
def add_userdata(name,username,password,email,usertype,date_reg):
    c.execute('INSERT INTO users(name, username,password,email,usertype,dateofreg) VALUES (?, ?, ?, ?, ?, ?)', (name, username, password, email, usertype,date_reg))
    conn.commit()

def view_all_users():
	c.execute('SELECT * FROM users')
	data = c.fetchall()
	return data

def view_all_tables():
	c.execute('SELECT * FROM sqlite_master')
	data = c.fetchall()
	return data

def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()


#container 2

con2 = st.container(height=400, border=True)
con2.subheader('Create user account')
name = con2.text_input('Fullname')
username = con2.text_input('Username')
password = con2.text_input('Password', type='password')
email = con2.text_input('Email')
usertype = con2.selectbox('User type', 
                          ('Reviewer','Admin'),
                          index=None,
                          placeholder="Select user type...")
date_reg = con2.date_input('Date of registration', disabled=True)
if con2.button('Save'):
    view_all_tables()
    create_user()
    if add_userdata(name,username,  make_hashes(password), email, usertype,date_reg):
        con2.success('user creation successful.')
        view_all_users()

