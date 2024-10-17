import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from streamlit.source_util import _on_pages_changed, get_pages
from dotenv import load_dotenv
from time import sleep
import os
from menu import menu
import hashlib
import json
from pathlib import Path
import sqlite3
import pandas as pd
from st_pages import add_page_title, get_nav_from_toml
from image_loader import render_image




st.set_page_config( initial_sidebar_state="collapsed")



conn = sqlite3.connect('data.db')
c = conn.cursor()



    
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
    


def main():
    load_dotenv()
    
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

st.markdown(page_bg_img, unsafe_allow_html=True)
#render_image(r"ghcoatofarms.png")
try:
    def create_user():
        c.execute("CREATE TABLE IF NOT EXISTS users(name TEXT, username TEXT, password TEXT,email TEXT,usertype TEXT, dateofreg TEXT)")
        
    def login_user(username,password):
        c.execute('SELECT * FROM "main.users" WHERE username=? AND password=?', (username, password))
        data = c.fetchall()
        return data

    def view_all_users():
        c.execute('SELECT * FROM "main.users"')
        data = c.fetchall()
        return data
except Exception as e:
  print(e)
  


col1, col2 = st.columns([2,1], gap="small")

col1.title(":orange[DocRev-AI]")
col1.write("**AI Assistant for Public Policy Reviewers in Ghana**")
col2.subheader("User login", divider=True)
username = col2.text_input('User name')
password = col2.text_input('Password', type='password')
loginButton = col2.button("login", key='login')



if loginButton:
  
  if username and password:
   
    result = login_user(username, password)
    if result:
        st.session_state["logged_in"] = True
        st.session_state["user"] = username
        
        sleep(0.5)
        

        if result[:4] =='Admin':
            st.session_state.user = 'Admin'
            nav = get_nav_from_toml(".streamlit/pages_sections.toml")
            #show_all_pages()
            st.switch_page('pages/admin.py')
        elif result[:4] =='Reviewer':
            st.session_state.user = 'Reviewer'
            nav = get_nav_from_toml(".streamlit/pages.toml")
            #show_all_pages()
            st.switch_page('pages/user.py')
        else:
            st.write('No result from DB')
    else:
        col2.error("Incorrect username/password",icon=":material/error:")
  else:
    col2.warning("Please enter your access details",icon=":material/warning:")  
        
        
if __name__ == '__main__':
    main()