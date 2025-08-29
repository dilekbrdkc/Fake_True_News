import streamlit as st
import pickle


with open("C:/Users/dilek/Desktop/FakeTrueNews/logistic_model.pkl", "rb") as model_file:
    loaded_model = pickle.load(model_file)

with open("C:/Users/dilek/Desktop/FakeTrueNews/tfidf_vectorizer.pkl", "rb") as vec_file:
    loaded_tfidf = pickle.load(vec_file)

st.title("Fake News Detection App")

user_input = st.text_area("Write News Text Here")

if st.button("Guess"):
    if user_input.strip() == "":
        st.warning("Please enter text!")
    else:
        vector = loaded_tfidf.transform([user_input])
        prediction = loaded_model.predict(vector)
        st.success(f"Estimation: **{prediction[0]}**")

# For Run The Code, Please Write to Terminal "streamlit run C:/Users/dilek/Desktop/FakeTrueNews/App.py

