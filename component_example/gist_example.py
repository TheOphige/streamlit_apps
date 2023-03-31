import streamlit as st
from streamlit_embedcode import github_gist

st.title("Github Gist Example")
st.write("Code from Palmer's Penguin Streamlit app.")

github_gist('https://gist.github.com/TheOphige/3eedd7ad0033d4d1b7f27d8cf5acab24')