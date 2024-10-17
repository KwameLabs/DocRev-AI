import streamlit as st
from pypdf import PdfReader
from menu import menu_with_redirect



# Redirect to app.py if not logged in, otherwise show the navigation menu
#menu_with_redirect() 
#st.set_page_config( initial_sidebar_state="collapsed")

if st.session_state["logged_in"] ==False:
    st.warning("You do not have permission to view this page.")
    st.stop()
    
import sqlite3
conn = sqlite3.connect('data.db')
c = conn.cursor()
# Verify the user's role

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


if st.session_state["logged_in"] == True:
    st.header('Logged in as {}'.format(st.session_state['user']))
    
        

con1 = st.container(height=300, border=True)
con1.write()

pdf_docs=con1.file_uploader("**Upload standards, national objectives, existing policies etc. PDF here and click on 'Process document'**", accept_multiple_files=True,  key='pdf')
processBtn = con1.button("Process document", type='secondary',)


#con1.markdown(page_bg_img, unsafe_allow_html=True)

#container 2

# if st.sidebar.link_button('Logout',url='#'):
#         st.session_state["logged_in"] = False
#         st.session_state["user"] = ''
#         st.switch_page('home.py')

# for key in st.session_state.keys():
#     del st.session_state[key]