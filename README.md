# Product Reviews Analysis System

The **Product Reviews Analysis System** is an AI-based desktop application that analyzes online product reviews and identifies whether they are **Genuine**, **Suspicious**, **Likely Fake**, or **Neutral**.  
The system supports both **manual text input** and **review screenshots**, making it useful for real-world review analysis across multiple e-commerce platforms.

This project is developed as an **educational AI project**, combining **Machine Learning**, **Natural Language Processing (NLP)**, **rule-based heuristics**, and **OCR**.

---

## 🎯 Key Objectives

- Detect fake or misleading product reviews
- Assist users in making informed purchasing decisions
- Demonstrate practical use of AI techniques in real-world problems
- Provide explainable results with confidence scores and visual insights

---

## 🔍 Key Features

- Manual review text analysis
- Screenshot-based review analysis using OCR
- Platform-specific handling for:
  - Amazon
  - Flipkart
  - Myntra
  - Nykaa
  - Meesho
- Hybrid detection approach:
  - ML model (TF-IDF + SGDClassifier)
  - Sentiment analysis (VADER)
  - Linguistic & behavioral rules
- Review classification:
  - Genuine
  - Suspicious
  - Likely Fake
  - Neutral
- Confidence score for every prediction
- Charts for visual insights using Matplotlib
- User-friendly GUI built using Tkinter

---

## 🧠 How the System Works

1. **Input**
   - User enters review text OR uploads review screenshots

2. **OCR Processing**
   - Screenshots are processed using PaddleOCR to extract text

3. **Preprocessing**
   - UI noise removal (helpful buttons, ratings, usernames)
   - Tokenization and lemmatization
   - Platform-specific review splitting

4. **Analysis Engine**
   - ML model prediction
   - Rule-based scoring (generic phrases, repetition, promotional tone)
   - Sentiment analysis

5. **Final Decision**
   - Hybrid score → classification + confidence percentage

6. **Visualization**
   - Summary chart
   - Average confidence chart

---

## 🛠️ Technologies Used

- **Programming Language:** Python  
- **GUI:** Tkinter  
- **Machine Learning:** Scikit-learn  
- **NLP:** NLTK, VADER Sentiment  
- **OCR:** PaddleOCR  
- **Visualization:** Matplotlib  
- **Data Handling:** Pandas, NumPy  

---

## 📁 Project Structure

```text
Product-Reviews-Analysis-System/
│
├── data/
│   ├── reviews.csv
│   └── fake_review_sample.csv
│
├── gui/
│   ├── main_gui.py
│   ├── charts.py
│   └── theme/azure/
│       ├── azure.tcl
│       └── theme/
│
├── logs/
│
├── models/
│   ├── review_model_final.pkl
│   └── vectorizer_final.pkl
│
├── scripts/
│   ├── generate_dataset.py
│   ├── ocr_reader.py
│   ├── predict_bulk.py
│   ├── predict_review.py
│   ├── preprocess.py
│   └── train_model.py
│
├── utils/
│   ├── logger.py
│   └── __init__.py
│
├── requirements.txt
├── README.md
└── .gitignore

#gmail ---- nagarkrrish907@gmail.com 
gmail for more details and dataset
