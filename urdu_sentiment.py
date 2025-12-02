import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

data = [
    ("یہ سروس بہت اچھی تھی", "pos"),
    ("مجھے یہ پروڈکٹ بالکل پسند نہیں آیا", "neg"),
    ("ڈلیوری وقت پر ہوئی، میں مطمئن ہوں", "pos"),
    ("کوالٹی خراب ہے", "neg"),
    ("قیمت مناسب اور تجربہ اچھا رہا", "pos"),
    ("تاخیر ہوئی اور سپورٹ بھی کمزور تھی", "neg"),
]
df = pd.DataFrame(data, columns=["text","label"])

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=1)),
    ("clf", LogisticRegression(max_iter=200))
])

Xtr, Xte, ytr, yte = train_test_split(df["text"], df["label"], test_size=0.4, random_state=42, stratify=df["label"])
pipe.fit(Xtr, ytr)
pred = pipe.predict(Xte)
print(classification_report(yte, pred))

samples = ["سروس بہترین ہے", "پیسوں کے ضیاع کے برابر"]
print(pipe.predict(samples))
