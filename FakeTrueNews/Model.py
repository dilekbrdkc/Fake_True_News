import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Veri setini oku
df_fake = pd.read_csv("C:/Users/dilek/Desktop/FakeTrueNews/Fake.csv")
df_true = pd.read_csv("C:/Users/dilek/Desktop/FakeTrueNews/True.csv")
df_fake["label"] = "FAKE"
df_true["label"] = "REAL"
df = pd.concat([df_fake, df_true]).sample(frac=1).reset_index(drop=True)
df = df[["text", "label"]].dropna()


X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.25, random_state=42)

# TF-IDF
tfidf = TfidfVectorizer(stop_words="english", max_df=0.7)
tfidf_train = tfidf.fit_transform(X_train)


lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(tfidf_train, y_train)


with open("C:/Users/dilek/Desktop/FakeTrueNews/logistic_model.pkl", "wb") as model_file:
    pickle.dump(lr_model, model_file)

# TF-IDF vector
with open("C:/Users/dilek/Desktop/FakeTrueNews/tfidf_vectorizer.pkl", "wb") as vec_file:
    pickle.dump(tfidf, vec_file)

#
with open("C:/Users/dilek/Desktop/FakeTrueNews/logistic_model.pkl", "rb") as model_file:
    loaded_model = pickle.load(model_file)

with open("C:/Users/dilek/Desktop/FakeTrueNews/tfidf_vectorizer.pkl", "rb") as vec_file:
    loaded_tfidf = pickle.load(vec_file)


sample_text = ["The president will visit the capital next week."]
vector = loaded_tfidf.transform(sample_text)
prediction = loaded_model.predict(vector)

print("Prediction:", prediction[0])  # FAKE or REAL
