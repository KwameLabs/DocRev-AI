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


# DEFAULT_PAGE = "home.py"
# SECOND_PAGE_NAME = "admin"

st.set_page_config( initial_sidebar_state="collapsed")

#st.session_state['login'] = True

# if st.session_state['login']:
    
#     nav = get_nav_from_toml(
#         ".streamlit/notsignin_user_menu.toml"
#     )
#     pg = st.navigation(nav)
#     #add_page_title(pg)
#     pg.run()

# Convert Pass into hash format
# def make_hashes(password):
#     return hashlib.sha256(str.encode(password)).hexdigest()


# # Check password matches during login
# def check_hashes(password, hashed_text):
#     if make_hashes(password) == hashed_text:
#         return hashed_text
#     return False


conn = sqlite3.connect('data.db')
c = conn.cursor()




# Convert Pass into hash format
# def make_hashes(password):
# 	return hashlib.sha256(str.encode(password)).hexdigest()

# # Check password matches during login
# def check_hashes(password,hashed_text):
# 	if make_hashes(password) == hashed_text:
# 		return hashed_text
# 	return False

    
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
    




#menu() # Render the dynamic menu!
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

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


def login_user(username,password):
    c.execute('SELECT * FROM users2 WHERE username=? AND password=?', (username, password))
    data = c.fetchall()
    return data

def view_all_users():
    c.execute("SELECT * FROM users2")
    data = c.fetchall()
    return data

# st.logo("images/horizontal_blue.png", icon_image="images/icon_blue.png")

#st.title("DocRev-AI")
# username = st.text_input('Username')
# password = st.text_input('Password', type='password')
#st.logo("gh_coat.png")
#col1, col2 = st.columns([2,1], gap="large")

#with col1:
    #render_image("images/DocRev-AI_header2.png")

render_image("images/DocRev-AI_header.png")
#col1.title(":orange[DocRev-AI]")
st.markdown("""
        <style>
        .big-font {
            font-size:18px !important;
            font-weight: bold !important;
            text-align: center;
        }
        </style>
        """, unsafe_allow_html=True)

st.markdown('<p class="big-font">AI Assistant for Public Policy Reviewers in Ghana</p>', unsafe_allow_html=True)

st.subheader("User login", divider=True)
username = st.text_input('User name')
password = st.text_input('Password', type='password')
loginButton = st.button("login", key='login')
#st.logo("images/horizontal_blue.png", icon_image="images/icon_blue.png")

# st.title(":orange[DocRev-AI]")
# st.write("**AI Assistant for Public Policy Reviewers in Ghana**")
# st.subheader("User login", divider=True)
# username = st.text_input('User name')
# password = st.text_input('Password', type='password')
# loginButton = st.button("login", key='login')
#file_1 = os.path.join(base_dir, "pages", "client_1.py")
#st.page_link(file_1, label="1", icon="📌")

if loginButton:
   # if password == 'admin':
   #hashed_pswd = make_hashes(password)
#    result = login_user(username, check_hashes(password, hashed_pswd))
   result = login_user(username, password)
   if result:
        st.session_state["logged_in"] = True
        st.session_state["user"] = username
        #col2.success("Logged in as {}".format(username))
        # if st.success:
        #             st.subheader("User Profiles")
        #             user_result = view_all_users()
        #             clean_db = pd.DataFrame(
        #                 user_result, columns=["Name","Username", "Password", "Email", "UserType", "Date of Registration"]
        #             )
        #             st.dataframe(clean_db)
        sleep(0.5)
        
        # if result[:4] =='Admin':
        #     nav = get_nav_from_toml(".streamlit/pages_sections.toml")
        # else:
        #     nav = get_nav_from_toml(".streamlit/pages.toml")

        if result[:4] =='Admin':
            st.session_state.user = 'Admin'
            nav = get_nav_from_toml(".streamlit/pages_sections.toml")
            #show_all_pages()
            st.switch_page('pages/admin.py')
        else:
            st.session_state.user = 'Reviewer'
            nav = get_nav_from_toml(".streamlit/pages.toml")
            #show_all_pages()
            st.switch_page('pages/user.py')
   else:
        st.error("Incorrect username/password",icon=":material/error:")
        
#    if st.session_state["logged_in"]:
        # show_all_pages()
        # hide_page(DEFAULT_PAGE.replace(".py", ""))
        # switch_page(SECOND_PAGE_NAME)
#    else:
        # clear_all_but_first_page()
       
if __name__ == '__main__':
    main()