import streamlit as st


def top_header():
    html = """
    <style>
             
        .header {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        background-color: #99ff99;
        padding: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: #011936;
        z-index: 1000;
        box-shadow: 0 4px 2px -2px gray;
        
    }
    
      .header {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        background-color: #1C3144;
        padding: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: #fff;
        z-index: 1000;
        box-shadow: 0 4px 2px -2px gray;
    }
    
    #MainMenu {visibility:hidden;}
    footer {visibility:hidden;}
    header {visibility:hidden;}
    
    /* Add some padding below the header to prevent content overlap */
    .main-content {
        margin-top: 70px;
    }
    
   .reportview-container .css-1lcbmhc .css-1outpf7 {{
                    padding-top: 35px;
                }}
                .reportview-container .main .block-container {{
                    {max_width_str}
                    padding-top: {padding_top}rem;
                    padding-right: {padding_right}rem;
                    padding-left: {padding_left}rem;
                    padding-bottom: {padding_bottom}rem;
                }}
                .reportview-container .main {{
                    color: {color};
                    background-color: {background_color};
                }}
    
    .block-container {
                    padding-top: 4rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                    
                }
                
    .rightsidebar {
        padding-top: 4rem;
        position: fixed;
        width: 200px;
        height: 400px;
        background: #000;
        margin-left: auto; 
        margin-right: 0;
    }
    
    [data-testid="stSidebarNav"] {{
    position:absolute;
    bottom: 0;
    z-index: 1;
    background: {color};
    }}
    [data-testid="stSidebarNav"] > ul {{
        padding-top: 2rem;
    }}
    [data-testid="stSidebarNav"] > div {{
        position:absolute;
        top: 0;
    }}
    [data-testid="stSidebarNav"] > div > svg {{
        transform: rotate(180deg) !important;
    }}
    [data-testid="stSidebarNav"] + div {{
        overflow: scroll;
        max-height: 66vh;
    }}
    
    div:has( >.element-container div.floating) {
    display: flex;
    flex-direction: column;
    position: fixed;
    }

    div.floating {
        height:0%;
    }
    </style>
    
    <div class="header">
        DocRev-AI: AI-Assisted Public Policy Review System for Ghana
    </div>
    
    """
    st.markdown(html, unsafe_allow_html=True)
    st.logo("images/DocRev-AI_Assistant.png", icon_image="images/DocRev-AI_logo.png")
    
def side_bar():
    st.markdown(
    """
<style>
.css-nzvw1x {
    background-color: #061E42 !important;
    background-image: none !important;
}
.css-1aw8i8e {
    background-image: none !important;
    color: #FFFFFF !important
}
.css-ecnl2d {
    background-color: #496C9F !important;
    color: #496C9F !important
}
.css-15zws4i {
    background-color: #496C9F !important;
    color: #FFFFFF !important
}
</style>
""",
    unsafe_allow_html=True
)