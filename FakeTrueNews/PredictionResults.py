import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


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
tfidf_test = tfidf.transform(X_test)


lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(tfidf_train, y_train)


y_pred = lr_model.predict(tfidf_test)


print("Doğruluk Oranı: %", round(accuracy_score(y_test, y_pred) * 100, 2))


results_df = pd.DataFrame({
    "text": X_test,
    "actual_label": y_test,
    "predicted_label": y_pred
})


results_df.to_csv("C:/Users/dilek/Desktop/FakeTrueNews/prediction_results.csv", index=False, header=True)

print("Prediction results are saved in the prediction_results.csv file.")
