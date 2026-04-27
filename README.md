# Fake News Detection using Streamlit

This project uses the Kaggle Fake News Detection dataset and a Machine Learning model
(TF-IDF + PassiveAggressiveClassifier) to classify news as Fake or Real.

## Dataset

Download the Kaggle dataset:
https://www.kaggle.com/c/fake-news/data

Place `train.csv` inside the project root folder.

Common columns:
- id
- title
- author
- text
- label

## Run Locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train model

```bash
python model_training.py
```

This creates:

```bash
model.pkl
```

### 3. Run Streamlit app

```bash
streamlit run app.py
```

## GitHub Ready Structure

```text
fake_news_detection_streamlit/
│
├── app.py
├── model_training.py
├── requirements.txt
├── README.md
├── .gitignore
└── train.csv
```
