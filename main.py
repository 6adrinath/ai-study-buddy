import streamlit as st
import os
import fitz  # PyMuPDF
import openai
import spacy

# Page config
st.set_page_config(page_title="AI Study Buddy (OpenRouter)", layout="wide")
st.title("📚 AI Study Buddy – Powered by OpenRouter (Mistral or GPT)")
st.write("Upload your notes and ask questions. Powered by cloud-based LLMs.")

# OpenRouter API key and base
openai.api_key = os.getenv("OPENROUTER_API_KEY")
openai.api_base = "https://openrouter.ai/api/v1"

# Ensure spaCy model is available
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Extract main points from notes
def extract_main_points(text, max_sentences=5):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 40]
    return sentences[:max_sentences]

# Upload notes
uploaded_file = st.file_uploader("Upload your notes (PDF or TXT)", type=["pdf", "txt"])
file_text = ""

if uploaded_file:
    if uploaded_file.type == "text/plain":
        file_text = uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "application/pdf":
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for page in doc:
            file_text += page.get_text()
    st.success("✅ Notes loaded!")

# Ask a question
if file_text:
    question = st.text_input("Ask a question from your notes:")
    if question:
        with st.spinner("🤖 Thinking..."):
            try:
                key_sentences = extract_main_points(file_text)
                refined_notes = "\n".join(key_sentences)

                prompt = f"""You are a helpful AI tutor. Based only on these notes, answer the student's question clearly and concisely.

Notes:
\"\"\"
{refined_notes}
\"\"\"

Question: {question}
"""

                response = openai.ChatCompletion.create(
                    model="mistralai/mistral-7b-instruct",  # You can change to gpt-3.5 etc
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=400,
                )

                answer = response['choices'][0]['message']['content']
                st.success("🎯 Answer:")
                st.write(answer)

            except Exception as e:
                st.error(f"❌ Error: {e}")
