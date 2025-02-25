# Python-Data-Analysis-and-Processing
# 🧠 AI & Machine Learning Demonstrations

## 📌 Overview
This repository contains three Python scripts showcasing **recent advancements in NLP (Natural Language Processing) and CV (Computer Vision)**, along with **an understanding of ML/DL (Machine Learning/Deep Learning) algorithms**.  
It covers:
- **NLP:** Sentiment analysis using a Transformer model.
- **CV:** Image classification using a pre-trained deep learning model.
- **ML/DL:** Training an autoencoder on the MNIST dataset.

---

## 📂 Files & Features

### 1️⃣ **NLP - Sentiment Analysis (`sentiment_analysis.py`)**  
📌 **Description:**  
- Uses **Hugging Face's Transformers** (`distilbert-base-uncased`) to analyze sentiment (positive/negative).  
- Processes sample text inputs and classifies their sentiment.

📌 **Example Usage:**
```python
from transformers import pipeline

nlp_pipeline = pipeline("sentiment-analysis")
print(nlp_pipeline("I love machine learning!")) 

