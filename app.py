
import streamlit as st
import joblib
import re
from pypdf import PdfReader

# Load trained Machine Learning model
model = joblib.load("knn_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Website settings
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="📄",
    layout="wide"
)

# Website heading
st.title("AI Resume Screening & Job Matching System")
st.write("Upload a resume and compare it with the job requirements.")

st.divider()

# Resume upload
st.subheader("1. Upload Candidate Resume")

uploaded_file = st.file_uploader(
    "Choose your resume",
    type=["pdf"]
)

# Job description input
st.subheader("2. Enter Job Description")

job_description = st.text_area(
    "Paste the job description here",
    height=180,
    placeholder="Example: Python, Machine Learning, SQL, Pandas..."
)

# Analyze button
if st.button("Analyze Resume", type="primary"):

    if uploaded_file is None:
        st.warning("Please upload a PDF resume.")

    elif not job_description.strip():
        st.warning("Please enter a job description.")

    else:
        # Read PDF resume
        pdf = PdfReader(uploaded_file)
        resume_text = ""

        for page in pdf.pages:
            resume_text += page.extract_text() or ""

        if not resume_text.strip():
            st.error("No readable text found in this PDF.")
            st.stop()

        # KNN prediction
        resume_vector = vectorizer.transform([resume_text])
        prediction = model.predict(resume_vector)[0]

        # Extract keywords
        stop_words = {
            "the", "and", "for", "with", "you", "your",
            "are", "our", "will", "have", "has", "from",
            "this", "that", "who", "what", "can", "all",
            "job", "role", "work", "working", "years",
            "year", "experience", "required", "skills",
            "ability", "knowledge", "good", "strong"
        }

        def extract_keywords(text):
            words = re.findall(r"[a-zA-Z+#.]+", text.lower())

            keywords = set()

            for word in words:
                if len(word) > 2 and word not in stop_words:
                    keywords.add(word)

            return keywords

        # Compare resume and job description
        resume_keywords = extract_keywords(resume_text)
        job_keywords = extract_keywords(job_description)

        matched_skills = resume_keywords.intersection(job_keywords)
        missing_skills = job_keywords - resume_keywords

        if len(job_keywords) > 0:
            match_percentage = (
                len(matched_skills) / len(job_keywords)
            ) * 100
        else:
            match_percentage = 0

        # Display results
        st.divider()
        st.header("3. Resume Analysis Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Job Keyword Match",
                f"{match_percentage:.1f}%"
            )

        with col2:
            st.metric(
                "Matched Keywords",
                len(matched_skills)
            )

        st.subheader("KNN Model Prediction")

        if prediction == "Suitable":
            st.success("Suitable")
        else:
            st.warning("Not Suitable")

        st.subheader("Matched Skills / Keywords")

        if matched_skills:
            st.write(", ".join(sorted(matched_skills)))
        else:
            st.write("No matching keywords found.")

        st.subheader("Missing Skills / Keywords")

        if missing_skills:
            st.write(", ".join(sorted(missing_skills)))
        else:
            st.success("All extracted job keywords are present!")

        st.caption(
            "Note: This is a prototype. KNN predicts using the "
            "training dataset. Keyword matching is a separate "
            "comparison, not a verified hiring decision."
        )
