import streamlit as st
import os
import fitz  # PyMuPDF
import openai
import spacy

# Setup
st.set_page_config(page_title="AI Study Buddy (OpenRouter)", layout="wide")
st.title("📚 AI Study Buddy – Powered by OpenRouter (GPT or Mistral)")
st.write("Upload your notes and ask questions using cloud-based LLMs.")

# Configure OpenRouter (OpenAI-compatible)
openai.api_key = os.getenv("OPENROUTER_API_KEY")
openai.api_base = "https://openrouter.ai/api/v1"

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def extract_main_points(text, max_sentences=5):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 40]
    return sentences[:max_sentences]

# File uploader
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

# Ask question
if file_text:
    question = st.text_input("Ask a question from your notes:")
    if question:
        with st.spinner("🤖 Thinking..."):
            try:
                # Extract main points
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
                    model="mistralai/mistral-7b-instruct",  # or "openai/gpt-3.5-turbo"
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=400,
                )
                answer = response['choices'][0]['message']['content']
                st.success("🎯 Answer:")
                st.write(answer)
            except Exception as e:
                st.error(f"❌ Error: {e}")
