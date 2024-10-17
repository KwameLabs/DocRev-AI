import streamlit as st

t = 'A faster way to build and share data apps'

html_style = '''
<style>
div:has( >.element-container div.floating) {
    display: flex;
    flex-direction: column;
    position: fixed;
}

div.floating {
    height:0%;
}
</style>
'''
st.markdown(html_style, unsafe_allow_html=True)

col1, col2 = st.columns([9, 2])

with col1:
    for i in range(0, 30):
        st.header("Today's news")
        st.write(t)

with col2:
    st.markdown(
        '''
            <div class="floating">
                <a href='https://streamlit.io/'>Hello Streamlit</a>
                <a href='https://streamlit.io/'>Hello Streamlit</a>
                <a href='https://streamlit.io/'>Hello Streamlit</a>
            </div>
        ''', 
        unsafe_allow_html=True
        )