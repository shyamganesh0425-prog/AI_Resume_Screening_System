# AI Resume Screening & Job Matching System

## About the Project
An AI-based resume screening application that uses Machine Learning to predict resume suitability and compare candidate resumes with job descriptions.

## Technologies Used
- Python
- Machine Learning
- K-Nearest Neighbors (KNN)
- TF-IDF Vectorization
- Streamlit
- Scikit-learn
- PyPDF

## Key Features
- Upload resumes in PDF format.
- Extract text from uploaded resumes.
- Predict resume suitability using KNN.
- Compare resume keywords with job description.
- Display matched and missing keywords.
- Show job keyword match percentage.

## How It Works
1. Upload a candidate resume.
2. Enter the job description.
3. Extract resume text using PyPDF.
4. Convert text into numerical features using TF-IDF.
5. Predict suitability using the trained KNN model.
6. Display keyword matching results.

## Run Locally

Install dependencies:

```bash
pip install streamlit scikit-learn pandas joblib pypdf
```

Run the application:

```bash
python -m streamlit run app.py
```

## Note
This is a prototype developed using a small sample dataset for learning and demonstration purposes. It is not intended to make actual hiring decisions.
