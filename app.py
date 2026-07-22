import streamlit as st

from model import predict

st.set_page_config(page_title="MCQ Solver")

st.title("🧠 Smart MCQ Solver")

question=st.text_area("Question")

A=st.text_input("Option A")

B=st.text_input("Option B")

C=st.text_input("Option C")

D=st.text_input("Option D")

E=st.text_input("Option E")

if st.button("Predict"):

    result=predict(question,A,B,C,D,E)

    st.subheader("Top Predictions")

    for label,prob in result:

        st.write(f"{label} : {prob:.4f}")
