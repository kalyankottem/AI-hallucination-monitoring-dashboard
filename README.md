# AI Hallucination Monitoring Dashboard for Business QA Systems

## Description
This project is an interactive dashboard for detecting hallucinated responses in AI-powered Question Answering / RAG systems by comparing generated answers against retrieved source documents.

It helps organizations monitor AI trustworthiness, detect unsupported claims, and improve response reliability.

---

## Problem Statement
AI-based QA systems may generate hallucinated responses that are incorrect or unsupported by retrieved source documents, leading to misinformation and reduced trust.

This project provides an automated hallucination monitoring dashboard for AI governance.

---

## Objective
- Capture user queries, AI responses, and retrieved documents  
- Compare generated responses with source evidence  
- Detect hallucinations / unsupported answers  
- Assign trust/confidence scores  
- Visualize grounding insights interactively  

---

## Features
- Grounded / Partially Grounded / Hallucinated Classification  
- Trust Score Calculation  
- Hallucination Distribution Visualization  
- Trust Score Histogram  
- Detailed Response Analysis Table  
- Flagged Hallucination Monitoring Table  

---

## Methodology
1. Collect QA logs from RAG system  
2. Preprocess AI responses and source documents  
3. Convert text into TF-IDF vectors  
4. Compute cosine similarity between response and evidence  
5. Classify grounding level  
6. Assign trust/confidence score  
7. Display results in dashboard  

---

## Technologies Used
- Python  
- Streamlit  
- Pandas  
- Scikit-learn  
- Plotly  

---

## Applications
- AI Governance  
- RAG Evaluation  
- Enterprise AI Monitoring  
- Hallucination Detection  

---

## Future Improvements
- Use semantic embeddings instead of TF-IDF  
- Add real-time RAG pipeline integration  
- Track hallucination trends over time  
- Add alerting for critical hallucinations  

---

## Run Locally
```bash
python -m streamlit run ai_hallucination_monitoring_dashboard.py
```
