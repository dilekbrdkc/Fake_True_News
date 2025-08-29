import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


fake_path = "C:/Users/dilek/Desktop/FakeTrueNews/Fake.csv"
true_path = "C:/Users/dilek/Desktop/FakeTrueNews/True.csv"


df_fake = pd.read_csv(fake_path)
df_true = pd.read_csv(true_path)


df_fake["label"] = "FAKE"
df_true["label"] = "REAL"

#
df = pd.concat([df_fake, df_true])
df = df.sample(frac=1).reset_index(drop=True)  

print(df.shape)
print(df.columns)
print(df["label"].value_counts())
df.head()

df = df[["text", "label"]]
df.dropna(inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.25, random_state=42
)

tfidf = TfidfVectorizer(stop_words="english", max_df=0.7)
tfidf_train = tfidf.fit_transform(X_train)
tfidf_test = tfidf.transform(X_test)

