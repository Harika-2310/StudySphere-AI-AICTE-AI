# study_buddy_app_gemini.py
import os
import json
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai  # Make sure: pip install google-generativeai


load_dotenv()
genai.configure(api_key="AIzaSyBM8yC1fOhrD4HeVNTbJ_iXYUsc_7CIYAY")


# --- Config / defaults ---
DEFAULT_MODEL = "gemini-2.5-flash-lite"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 400

# Configure Gemini API key
def set_gemini_api_key(key: str | None):
    if key:
        genai.configure(api_key=key)
    else:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))

# --- Gemini chat wrapper ---
def call_gemini_chat(messages, model="gemini-2.5-flash-lite", temperature=0.2, max_tokens=800):
    """
    messages: list of dicts with "role" and "content"
    """
    prompt_parts = []
    for m in messages:
        role = m.get("role", "user").upper()
        content = m.get("content", "")
        prompt_parts.append(f"[{role}]\n{content}\n")
    prompt = "\n".join(prompt_parts)

    # Use GenerativeModel and generate_content
    generative_model = genai.GenerativeModel(model)
    response = generative_model.generate_content(
        prompt,
        generation_config={
            "temperature": temperature,
            "max_output_tokens": max_tokens
        }
    )
    # Get the text output
    return response.text 
# --- Helpers ---
def try_parse_json(raw: str):
    try:
        return json.loads(raw)
    except Exception:
        import re
        m = re.search(r"(\{.*\}|\[.*\])", raw, re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                return None
        return None

def render_bullets(text: str):
    for line in text.splitlines():
        s = line.strip()
        if s:
            st.write("- " + s)

# --- Streamlit layout ---
st.set_page_config(page_title="AI Study Buddy (Gemini)", layout="wide")
st.title("📚 StudySphere AI")

# Sidebar: API key and settings
st.sidebar.header("Settings")
env_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
api_key = st.sidebar.text_input("Gemini API Key (optional)", value=env_key, type="password")
set_gemini_api_key(api_key or None)

model = st.sidebar.selectbox("Model", options=[DEFAULT_MODEL, "gemini-2.5-flash-lite"], index=0)
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=DEFAULT_TEMPERATURE, step=0.05)

# Tool selection
tool = st.radio("Choose tool", ["Explain", "Summarize", "Quiz", "Flashcards"], horizontal=True)
st.markdown("---")

# -----------------------------
# EXPLAIN TOOL
# -----------------------------
if tool == "Explain":
    st.header("Explain a topic (step-by-step)")
    with st.form("explain_form"):
        topic = st.text_input("Topic", placeholder="e.g., Convolutional Neural Networks")
        target_level = st.selectbox("Target level", ["beginner", "high school", "college", "advanced"])
        include_examples = st.checkbox("Include example(s)", value=True)
        submit = st.form_submit_button("Explain")

    if submit and topic.strip():
        with st.spinner("Generating explanation..."):
            system = {"role": "system", "content": "You are a helpful tutor who explains concepts in simple, student-friendly language."}
            examples_text = "Include one or two concrete examples." if include_examples else "Do not include examples."
            user = {"role": "user",
                    "content": f"Explain the following topic for a {target_level} learner.\nTopic: {topic}\n{examples_text}\nLimit response to ~300-450 words in short paragraphs or bullets."}
            try:
                raw = call_gemini_chat([system, user], model=model, temperature=temperature, max_tokens=600)
                st.subheader("Explanation")
                st.write(raw)
                st.download_button("Download explanation (.txt)", data=raw, file_name=f"explain_{topic[:30]}.txt")
            except Exception as e:
                st.error(f"API error: {e}")

# -----------------------------
# SUMMARIZE TOOL
# -----------------------------
elif tool == "Summarize":
    st.header("Summarize notes / lecture text into study bullets")
    with st.form("summarize_form"):
        text = st.text_area("Paste lecture notes / article", height=300)
        max_bullets = st.slider("Max bullets", min_value=3, max_value=20, value=8)
        submit = st.form_submit_button("Summarize")

    if submit and text.strip():
        with st.spinner("Summarizing..."):
            system = {"role": "system", "content": "You are an expert note-summarizer. Convert long text into concise study bullets."}
            user = {"role": "user",
                    "content": f"Summarize the text below into at most {max_bullets} clear study bullets.\nText:\n{text}"}
            try:
                raw = call_gemini_chat([system, user], model=model, temperature=temperature, max_tokens=600)
                bullets = [line.strip().lstrip("0123456789.)- ") for line in raw.splitlines() if line.strip()][:max_bullets]
                if bullets:
                    st.subheader("Study bullets")
                    for i, b in enumerate(bullets, 1):
                        st.markdown(f"**{i}.** {b}")
                    st.download_button("Download bullets (.txt)", data="\n".join(f"{i}. {b}" for i,b in enumerate(bullets,1)), file_name="summary_bullets.txt")
                else:
                    st.write(raw)
            except Exception as e:
                st.error(f"API error: {e}")

# -----------------------------
# QUIZ TOOL
# -----------------------------
elif tool == "Quiz":
    st.header("Generate Multiple-Choice Quiz (MCQs)")
    with st.form("quiz_form"):
        source_text = st.text_area("Paste notes or topic (short paragraph accepted)", height=250)
        topic = st.text_input("Optional topic")
        num_q = st.slider("Number of questions", min_value=1, max_value=10, value=5)
        submit = st.form_submit_button("Generate Quiz")

    if submit and (source_text.strip() or topic.strip()):
        prompt_source = source_text if source_text.strip() else f"Topic: {topic}"
        with st.spinner("Generating quiz..."):
            system = {"role": "system", "content": "You create short multiple-choice quizzes for students. Each question should have 4 options and exactly one correct answer."}
            user = {"role": "user",
                    "content": f"Create {num_q} multiple-choice questions (4 options each) based on the following content. Return JSON array: {{'question':'...','options':[...],'answer_index':0}}.\nContent:\n{prompt_source}"}
            try:
                raw = call_gemini_chat([system, user], model=model, temperature=temperature, max_tokens=700)
                parsed = try_parse_json(raw)
                if not parsed:
                    st.warning("Couldn't parse JSON reliably; showing raw AI output below.")
                    st.write(raw)
                else:
                    st.subheader("Quiz")
                    for i, q in enumerate(parsed, 1):
                        st.markdown(f"**Q{i}. {q.get('question','')}**")
                        for j, opt in enumerate(q.get("options", [])):
                            label = chr(65+j)
                            if j == q.get("answer_index", -1):
                                st.markdown(f"- **{label}. {opt}** ✅")
                            else:
                                st.markdown(f"- {label}. {opt}")
                    st.download_button("Download quiz as JSON", data=json.dumps(parsed, indent=2), file_name="quiz.json")
            except Exception as e:
                st.error(f"API error: {e}")

# -----------------------------
# FLASHCARDS TOOL
# -----------------------------
elif tool == "Flashcards":
    st.header("Generate flashcards (term → definition)")
    with st.form("flash_form"):
        source_text = st.text_area("Paste notes or topic", height=300)
        num_cards = st.slider("Number of flashcards", min_value=3, max_value=30, value=10)
        submit = st.form_submit_button("Generate Flashcards")

    if submit and source_text.strip():
        with st.spinner("Generating flashcards..."):
            system = {"role":"system","content":"You create concise study flashcards (term -> short definition)."}
            user = {
    "role": "user",
    "content": (
        f"From the content below, extract up to {num_cards} key terms or concepts "
        f"and give a one-sentence definition for each. "
        f"Return JSON array: [{{'term':'...','definition':'...'}}].\n"
        f"Content:\n{source_text}"
    )
}

            try:
                raw = call_gemini_chat([system, user], model=model, temperature=temperature, max_tokens=700)
                parsed = try_parse_json(raw)
                if not parsed:
                    st.warning("Couldn't parse JSON reliably; showing raw AI output below.")
                    st.write(raw)
                else:
                    st.subheader("Flashcards")
                    cols = st.columns(2)
                    for i, c in enumerate(parsed):
                        with cols[i % 2]:
                            st.markdown(f"**{c.get('term','')}**")
                            st.write(c.get("definition",""))
                    st.download_button("Download flashcards (JSON)", data=json.dumps(parsed, indent=2), file_name="flashcards.json")
            except Exception as e:
                st.error(f"API error: {e}")

# Footer
st.markdown("---")
st.write("Tip: tweak the temperature in the sidebar for more creative (higher) or more precise (lower) outputs.")
