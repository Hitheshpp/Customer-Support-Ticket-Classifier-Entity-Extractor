# Cell 1: Imports
import pandas as pd
import numpy as np
import re
import string
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Cell 2: Load Dataset
df = pd.read_excel(r"D:\customer_support_ticket\ai_dev_assignment_tickets_complex_1000.xls")
df.dropna(subset=["ticket_text", "issue_type", "urgency_level", "product"], inplace=True)

# Cell 3: Preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)

df["clean_text"] = df["ticket_text"].apply(clean_text)
df["processed_text"] = df["clean_text"].apply(preprocess)

# Cell 4: Feature Engineering
df["sentiment"] = df["processed_text"].apply(lambda x: TextBlob(x).sentiment.polarity)
df["ticket_length"] = df["processed_text"].apply(lambda x: len(x.split()))

urgency_keywords = ["urgent", "asap", "immediately", "soon", "waiting", "delay", "important", "priority", "now"]

def urgency_score(text):
    text = text.lower()
    return sum(word in text for word in urgency_keywords)

df["urgency_score"] = df["processed_text"].apply(urgency_score)

# Cell 5: TF-IDF
tfidf = TfidfVectorizer(max_features=500)
X_tfidf = tfidf.fit_transform(df["processed_text"])

# Cell 6: Final Feature Matrix
X = np.hstack((X_tfidf.toarray(), df[["sentiment", "ticket_length", "urgency_score"]].values))
y_issue = df["issue_type"]
y_urgency = df["urgency_level"]

X_train, X_test, y_train_issue, y_test_issue, y_train_urgency, y_test_urgency = train_test_split(
    X, y_issue, y_urgency, test_size=0.2, random_state=42
)

# Cell 7: Model Training
issue_model = LogisticRegression(max_iter=1000, class_weight='balanced')
urgency_model = RandomForestClassifier(n_estimators=100, random_state=42)

issue_model.fit(X_train, y_train_issue)
urgency_model.fit(X_train, y_train_urgency)

# Cell 8: Evaluation
print("Issue Type Classification Report:")
print(classification_report(y_test_issue, issue_model.predict(X_test)))

print("Urgency Level Classification Report:")
print(classification_report(y_test_urgency, urgency_model.predict(X_test)))

# Cell 9: Entity Extraction
product_list = df["product"].unique().tolist()
complaint_keywords = ["broken", "late", "error", "not working", "failed", "slow", "crash", "delay"]

def extract_entities(text):
    text_lower = text.lower()
    return {
        "products": [prod for prod in product_list if prod.lower() in text_lower],
        "dates": re.findall(r'\b(?:\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\b', text_lower),
        "complaints": [kw for kw in complaint_keywords if kw in text_lower]
    }

# Cell 10: Inference Function
def predict_ticket(ticket_text):
    cleaned = preprocess(clean_text(ticket_text))
    tfidf_vec = tfidf.transform([cleaned])
    length = len(cleaned.split())
    sentiment = TextBlob(cleaned).sentiment.polarity
    urgency_kw_score = urgency_score(cleaned)

    combined = np.hstack((
        tfidf_vec.toarray(),
        np.array([[sentiment, length, urgency_kw_score]])
    ))

    issue = issue_model.predict(combined)[0]
    urgency = urgency_model.predict(combined)[0]
    entities = extract_entities(ticket_text)

    return {
        "issue_type": issue,
        "urgency_level": urgency,
        "entities": entities
    }

# Cell 11: Test Prediction
test_input = "My device is not working and I need a replacement urgently. Please respond asap!"
result = predict_ticket(test_input)
print(json.dumps(result, indent=2))

# Cell 12: Gradio Interface
import gradio as gr

def gradio_interface(text):
    result = predict_ticket(text)
    return f"Issue: {result['issue_type']}\nUrgency: {result['urgency_level']}\nEntities: {json.dumps(result['entities'], indent=2)}"

demo = gr.Interface(fn=gradio_interface, inputs="textbox", outputs="text", title="Ticket Classifier")
demo.launch()
