import streamlit as st
import pandas as pd
import re
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# PAGE CONFIG
st.set_page_config(
    page_title="AI Hallucination Monitoring Dashboard",
    layout="wide"
)

st.title("🧠 AI Hallucination Monitoring Dashboard")

# SAMPLE RAG QA LOGS
data = {
    "User Query": [
        "What is the refund policy?",
        "How long is the warranty period?",
        "Can I cancel my subscription anytime?",
        "What countries do you ship to?"
    ],
    
    "AI Response": [
        "Customers can request refunds within 30 days of purchase.",
        "The warranty period is 5 years for all products.",
        "Yes, subscriptions can be cancelled anytime without penalty.",
        "We currently ship to over 100 countries worldwide."
    ],
    
    "Retrieved Document": [
        "Refunds are available within 30 days from purchase date.",
        "Warranty coverage is valid for 2 years from date of purchase.",
        "Subscriptions may be cancelled at any time through account settings.",
        "Shipping is available in 35 countries across North America and Europe."
    ]
}

df = pd.DataFrame(data)

# PREPROCESS FUNCTION
def preprocess(text):
    
    text = text.lower()
    
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    return text

# HALLUCINATION ANALYSIS
labels = []
scores = []

for _, row in df.iterrows():
    
    texts = [
        preprocess(row["AI Response"]),
        preprocess(row["Retrieved Document"])
    ]
    
    vectorizer = TfidfVectorizer()
    
    vecs = vectorizer.fit_transform(texts)
    
    similarity = cosine_similarity(
        vecs[0:1],
        vecs[1:2]
    )[0][0]
    
    scores.append(round(similarity, 2))
    
    if similarity >= 0.75:
        labels.append("Grounded")
    elif similarity >= 0.40:
        labels.append("Partially Grounded")
    else:
        labels.append("Hallucinated")

df["Trust Score"] = scores
df["Label"] = labels

# DASHBOARD METRICS
col1, col2, col3 = st.columns(3)

col1.metric("Total Responses", len(df))
col2.metric("Hallucinated", (df["Label"] == "Hallucinated").sum())
col3.metric("Average Trust Score", round(df["Trust Score"].mean(), 2))

# LABEL DISTRIBUTION
label_counts = df["Label"].value_counts()

fig1 = px.pie(
    values=label_counts.values,
    names=label_counts.index,
    title="Hallucination Classification Distribution"
)

st.plotly_chart(fig1, use_container_width=True)

# TRUST SCORE DISTRIBUTION
fig2 = px.histogram(
    df,
    x="Trust Score",
    nbins=10,
    title="Trust Score Distribution"
)

st.plotly_chart(fig2, use_container_width=True)

# DETAILED ANALYSIS TABLE
st.subheader("📄 Response Grounding Analysis")

st.dataframe(df)

# FLAGGED HALLUCINATIONS
st.subheader("⚠ Potential Hallucinations")

st.dataframe(
    df[df["Label"] == "Hallucinated"]
)