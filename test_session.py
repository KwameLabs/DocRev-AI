import streamlit as st

# Make list of random cat placeholder images
random_images = [
    "https://placekitten.com/200/400",
    "https://placekitten.com/300/400",
    "https://placekitten.com/400/400",
    "https://placekitten.com/500/400",
    "https://placekitten.com/600/400",
]

if "image_idx" not in st.session_state:
    st.session_state.image_idx = 0

if "image_ratings" not in st.session_state:
    st.session_state.image_ratings = {}

# Display the image
st.image(random_images[st.session_state.image_idx])

if st.session_state["image_idx"] in st.session_state.image_ratings:
    rating = st.session_state.image_ratings[st.session_state.image_idx]
else:
    rating = ""

with st.form("image_rating"):
    rating = st.text_input("Rate image", value=rating)

    if st.form_submit_button("Submit"):
        st.session_state.image_ratings[st.session_state.image_idx] = rating
        st.info("Submitted!")

if st.button("Next image", key="next"):
    idx = st.session_state.image_idx
    idx = (idx + 1) % len(random_images)
    st.session_state.image_idx = idx
    st.experimental_rerun()

elif st.button("Previous image", key="previous"):
    idx = st.session_state.image_idx
    idx = (idx - 1) % len(random_images)
    st.session_state.image_idx = idx
    st.experimental_rerun()

st.session_state