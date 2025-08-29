from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


import pandas as pd
df_fake = pd.read_csv("C:/Users/dilek/Desktop/FakeTrueNews/Fake.csv")
df_true = pd.read_csv("C:/Users/dilek/Desktop/FakeTrueNews/True.csv")
df_fake["label"] = "FAKE"
df_true["label"] = "REAL"
df = pd.concat([df_fake, df_true])
df = df.sample(frac=1).reset_index(drop=True)
df = df[["text", "label"]].dropna()


X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.25, random_state=42)


tfidf = TfidfVectorizer(stop_words="english", max_df=0.7)
tfidf_train = tfidf.fit_transform(X_train)
tfidf_test = tfidf.transform(X_test)


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

nb_model = MultinomialNB()
nb_model.fit(tfidf_train, y_train)
y_pred_nb = nb_model.predict(tfidf_test)
print(" Naive Bayes Accuracy Rate: %", round(accuracy_score(y_test, y_pred_nb) * 100, 2))

def plot_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=["FAKE", "REAL"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["FAKE", "REAL"], yticklabels=["FAKE", "REAL"])
    plt.xlabel("Estimate")
    plt.ylabel("Real")
    plt.title(f"Confusion Matrix - {title}")
    plt.show()

plot_confusion(y_test, y_pred_nb, "Naive Bayes")


