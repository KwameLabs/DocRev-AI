import streamlit as st
import openai
import time
import requests
from streamlit_lottie import st_lottie
from apikey import apikey
import os

#OpenAI key
os.environ['OPENAI_API_KEY'] = apikey



# Function to load Lottie animation from a URL
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Function to display Lottie animation while waiting for LLM results
def show_loading_animation():
    lottie_url = "https://assets5.lottiefiles.com/packages/lf20_z9ed2jna.json"  # Example loading animation
    lottie_json = load_lottieurl(lottie_url)
    if lottie_json:
        st_lottie(lottie_json, height=300, key="loading_animation")

# Function to call OpenAI API and get a response
def invoke_openai_llm(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100
    )
    return response.choices[0].text.strip()

# Main app function
def main():
    st.title("Policy Formulation Standard Compliance App")

    # User input
    user_input = st.text_area("Enter the policy document for analysis", height=200)

    if st.button("Analyze Policy"):
        if user_input:
            # Show the loading animation while processing
            with st.spinner("Analyzing with OpenAI..."):
                show_loading_animation()

                # Simulate a delay to mimic processing time
                time.sleep(2)

                # Call OpenAI LLM and get the result
                result = invoke_openai_llm(user_input)

                # Display the result
                st.success("Analysis Complete!")
                st.write("Result:", result)
        else:
            st.warning("Please enter some text to analyze.")

if __name__ == "__main__":
    main()
