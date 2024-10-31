import streamlit as st
from common import set_page_container_style
from header import top_header


st.set_page_config(
    page_title="Public Policy Document Review System",
    page_icon="👀",
    layout="wide",
    initial_sidebar_state="expanded",
    )

set_page_container_style(
        max_width = 1100, max_width_100_percent = True,
        padding_top = 0, padding_right = 10, padding_left = 5, padding_bottom = 10
)

top_header()

if "role" not in st.session_state:
    st.session_state.role = None
    
if "user" not in st.session_state:
    st.session_state.user = None

ROLES = [None, "Reviewer", "Admin"]
USERS = ['user1','user2','user3','user4','user5']
password = '1234'


def login():

    #st.header("DocRev-AI")
    #role = st.selectbox("Choose your role", ROLES)
    role = "Reviewer"
    
    #if st.session_state.user is None:
    col1, col2, col3 = st.columns([3,2,1], gap="small")
    

   # with col1:
        #render_image("images/DocRev-AI_header2.png")

    # render_image("images/DocRev-AI_header2.png")
    

   # st.markdown('<p class="big-font">AI Assistant for Public Policy Reviewers in Ghana</p>', unsafe_allow_html=True)
    with col1:
        st.image('images/rafiki2.png') 

    with col2:
        st.markdown("")
        st.markdown("")
        st.subheader(":blue[Welcome to DocRev-AI]")
        st.markdown("""
        <style>
        .big-font {
            font-size:18px !important;
            font-weight: bold !important;
            text-align: left;
        }
        .small-font {
            font-size:12px !important;
            font-weight: bold !important;
            text-align: center;
        }
        </style>
        """, unsafe_allow_html=True)
        st.subheader("User login", divider=True)
        username = st.text_input('User name')
        password = st.text_input('Password', type='password')
        #loginButton = st.button("login", key='login')

        if st.button("Log in"):
            
            if username in USERS and password == '1234':
                st.session_state.role = role
                st.session_state.user = username
                st.rerun()
            else:
                st.error('Username or password is incorrect')


def logout():
    st.session_state.role = None
    st.session_state.user = None
    st.rerun()


role = st.session_state.role

userguide = st.Page("userguide.py", title="User Guide", icon=":material/settings:")
logout_page = st.Page(logout, title="Log out", icon=":material/logout:")

review = st.Page(
    "reviewer/review.py",
    title="Review Document",
    icon=":material/help:",
    default=(role == "Reviewer"),
)

admin_1 = st.Page(
    "admin/create_user_account.py",
    title="Create User Account",
    icon=":material/person_add:",
    default=(role == "Admin"),
)



admin_2 = st.Page("admin/manage_user.py", title="Manage User", icon=":material/security:")

account_pages = [userguide, logout_page ]
review_pages = [review]
admin_pages = [admin_1, admin_2]

#st.title("DocRev-AI")


page_dict = {}
if st.session_state.role in ["Reviewer", "Admin"]:
    page_dict["Reviewer"] = review_pages
    
# if st.session_state.role == "Reviewer":
#     page_dict["Reviewer"] = review_pages
if st.session_state.role == "Admin":
    page_dict["Admin"] = admin_pages

if len(page_dict) > 0:
    pg = st.navigation({"Main Menu": account_pages} | page_dict)
else:
    pg = st.navigation([st.Page(login)])

   
pg.run()
