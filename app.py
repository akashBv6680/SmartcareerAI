import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import io
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai

# Optional TTS engines
try:
    import edge_tts
except Exception:
    edge_tts = None
try:
    from gtts import gTTS
except Exception:
    gTTS = None

# --- CONFIGURATION ---
LANGUAGE_DICT = {
    "English": "en",
    "Spanish": "es",
    "Arabic": "ar",
    "French": "fr",
    "German": "de",
    "Hindi": "hi",
    "Tamil": "ta",
    "Bengali": "bn",
    "Japanese": "ja",
    "Korean": "ko",
    "Russian": "ru",
    "Chinese (Simplified)": "zh-Hans",
    "Portuguese": "pt",
    "Italian": "it",
    "Dutch": "nl",
    "Turkish": "tr"
}
DEFAULT_LANGUAGE = "English"

def setup_llm():
    try:
        api_key = os.environ.get('GEMINI_API_KEY') or st.secrets.get('GEMINI_API_KEY')
        if not api_key:
            st.error("🔑 Error: GEMINI_API_KEY not found. Please set it in environment variables or Streamlit secrets.")
            return None
        return genai.Client(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize Gemini Client: {e}")
        return None

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_data():
    try:
        courses_df = pd.read_csv('courses.csv')
    except FileNotFoundError:
        st.error("Error: 'courses.csv' not found. Please create the file with required columns.")
        st.stop()
    courses_df.columns = courses_df.columns.str.strip().str.lower()
    if 'skill tags' in courses_df.columns:
        courses_df = courses_df.rename(columns={'skill tags': 'skill_tags'})
    model = load_model()
    courses_df['search_text'] = (
        courses_df['title'] + " " + courses_df['skill_tags'] + " " +
        courses_df['provider'] + " " + courses_df['level'] + " " +
        courses_df['prerequisites'] + " " + courses_df['duration']
    ).fillna('')
    course_embeddings = model.encode(courses_df['search_text'].tolist(), show_progress_bar=False)
    return courses_df, course_embeddings

def generate_user_embedding(user_profile, model):
    profile_text = (
        f"Education: {user_profile['education_level']} in {user_profile['major']}. "
        f"Technical Skills: {user_profile['technical_skills']}. "
        f"Soft Skills: {user_profile['soft_skills']}. "
        f"Goal Domain: {user_profile['target_domain']}."
    )
    return model.encode([profile_text])[0].reshape(1, -1)

def generate_llm_rationale(client, user_profile, course_row, timeline_type):
    if not client:
        return f"LLM Rationale Unavailable. Heuristic: Good fit for {course_row['level']} level. It is a {timeline_type} step."
    prompt = f"""Act as a highly experienced Career Counselor....
    User Profile: Education: {user_profile['education_level']} in {user_profile['major']}
    Skills: {user_profile['technical_skills']}
    Goal: Career switch/growth into {user_profile['target_domain']}
    Course Recommended: Title: {course_row['title']} by {course_row['provider']}
    Level: {course_row['level']} ({timeline_type} plan)
    Key Skills Taught: {course_row['skill_tags']}
    Prerequisites: {course_row['prerequisites']}
    Sentence 1 (Matching): Explain which existing user skills connect to the course content.
    Sentence 2 (Gap/Next Step): Explain what new, specific skill or knowledge gap this course fills for the user's target domain.
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config={"temperature": 0.3}
        )
        return response.text.strip().replace('\n', ' ')
    except Exception as e:
        return f"Error generating LLM rationale: {e}"

def recommend_courses(user_profile, courses_df, course_embeddings, model, llm_client):
    user_embed = generate_user_embedding(user_profile, model)
    similarity_scores = cosine_similarity(user_embed, course_embeddings)[0]
    results_df = courses_df.copy()
    results_df['similarity_score'] = similarity_scores

    # Prerequisite and Level matching, scoring, filtering - same logic as your code
    # ... (add all your filtering, mapping, scoring logic here unchanged)
    # For brevity, the rest remains as in your original app.

    # Generate LLM rationale
    # ... unchanged, just call generate_llm_rationale as needed
    # Return recommendations df
    ...

def get_rag_context(query, courses_df, course_embeddings, model, top_k=5):
    query_embed = model.encode([query])[0].reshape(1, -1)
    similarity_scores = cosine_similarity(query_embed, course_embeddings)[0]
    top_indices = np.argsort(similarity_scores)[::-1][:top_k]
    context = ""
    for i in top_indices:
        row = courses_df.iloc[i]
        context += (
            f"Course Title: {row['title']}, Provider: {row['provider']}, "
            f"Level: {row['level']}, Duration: {row['duration']}, "
            f"Skills: {row['skill_tags']}, Prerequisites: {row['prerequisites']} \n"
        )
    return context.strip()

def run_rag_query(query, courses_df, course_embeddings, model, llm_client):
    if not llm_client:
        return "The AI Agent is not initialized. Please ensure your Gemini API key is configured correctly."

    context = get_rag_context(query, courses_df, course_embeddings, model)
    rag_prompt = f"""You are the **PersonalAI Course Recommender** chatbot....
    User Query: "{query}"
    Context (Relevant Courses): ---
    {context}
    ---
    Based on the context, provide a concise, helpful answer. If the context does not contain the answer, state that you cannot find the information in the current catalog."""
    try:
        response = llm_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=rag_prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"Error communicating with the Gemini model: {e}"

def text_to_speech_conversion(text, lang_code, engine="gtts"):
    try:
        if engine == "edge_tts" and edge_tts is not None:
            tts = edge_tts.Communicate(text, lang=lang_code)
            audio_bytes = b""
            for chunk in tts.stream():
                audio_bytes += chunk
            return io.BytesIO(audio_bytes)
        elif engine == "gtts" and gTTS is not None:
            tts = gTTS(text=text, lang=lang_code)
            mp3_fp = io.BytesIO()
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            return mp3_fp
        else:
            st.warning("No TTS engine available for audio playback.")
            return None
    except Exception as e:
        st.warning(f"TTS Error: Could not generate speech with code '{lang_code}'. Try a different voice. Details: {e}")
        return None

# --- STREAMLIT UI CODE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "tts_enabled" not in st.session_state:
    st.session_state.tts_enabled = False
if "tts_language" not in st.session_state:
    st.session_state.tts_language = DEFAULT_LANGUAGE
if "tts_engine" not in st.session_state:
    st.session_state.tts_engine = "gtts"

st.set_page_config(layout="wide", page_title="AI Learning Path Recommender")
st.title("💡 AI-Powered Personalized Learning Path Recommender")
try:
    COURSES_DF, COURSE_EMBEDDINGS = load_data()
    MODEL = load_model()
    LLM_CLIENT = setup_llm()
except Exception as e:
    st.error(f"Could not initialize system components: {e}.")
try:
    with open('profiles.json', 'r') as f:
        SAMPLE_PROFILES = json.load(f)
except FileNotFoundError:
    SAMPLE_PROFILES = {}

col_input, col_output = st.columns([1, 2.5])

with col_input:
    st.header("👤 User Profile Input")
    profile_keys = ["Manual Input"] + list(SAMPLE_PROFILES.keys())
    profile_selection = st.selectbox("Load Sample Profile:", profile_keys)
    loaded_profile = SAMPLE_PROFILES[profile_selection] if profile_selection != "Manual Input" else {}
    st.subheader("Required Background")
    education_options = ["Bachelor's", "Master's", "PhD", "High School/GED", "Certificate"]
    education_level = st.selectbox("Education Level:", education_options, index=education_options.index(loaded_profile.get('education_level', "Bachelor's")))
    major = st.text_input("Major/Degree:", value=loaded_profile.get('major', "Computer Science"))
    technical_skills = st.text_area("Technical Skills (comma separated):", value=loaded_profile.get('technical_skills', "Python, SQL, Data Analysis, Excel, Git"))
    soft_skills = st.text_area("Soft Skills (comma separated):", value=loaded_profile.get('soft_skills', "Communication, Leadership, Problem-Solving"))
    st.subheader("Goals & Preferences (Optional)")
    target_domain = st.text_input("Target Career Domain (e.g., Data Science, UX Design, DevOps):", value=loaded_profile.get('target_domain', "Data Science"))
    duration_options = ["Short-term (1-3 months)", "Long-term (3-12 months)", "Any"]
    loaded_duration = loaded_profile.get('preferred_duration', "Any")
    duration_index = next((i for i, opt in enumerate(duration_options) if loaded_duration in opt), duration_options.index("Any"))
    preferred_duration = st.selectbox("Preferred Study Duration:", duration_options, index=duration_index)
    USER_PROFILE = {
        'education_level': education_level,
        'major': major,
        'technical_skills': technical_skills,
        'soft_skills': soft_skills,
        'target_domain': target_domain,
        'preferred_duration': preferred_duration,
    }
    st.markdown("---")
    if st.button("🚀 Generate Learning Path", type="primary"):
        st.session_state['path_generated'] = True
        st.session_state.messages = []
    st.subheader("🗣️ PersonalAI Chat Settings")
    st.session_state.tts_enabled = st.checkbox("Enable Text-to-Speech (TTS) Reply", value=st.session_state.tts_enabled)
    if st.session_state.tts_enabled:
        selected_lang_name = st.selectbox("Select Voice Language:", list(LANGUAGE_DICT.keys()), index=list(LANGUAGE_DICT.keys()).index(st.session_state.tts_language) if st.session_state.tts_language in LANGUAGE_DICT else 0)
        st.session_state.tts_language = selected_lang_name
        st.session_state.tts_engine = st.selectbox("TTS Engine", ["gtts", "edge_tts"], index=0 if gTTS else 1)

with col_output:
    st.markdown("## 🧠 Recommendation and Chat Output")
    # ... (main recommendations and conversation logic, unchanged except TTS section below)
    # In chat/assistant display, just update TTS logic to:
    # lang_code = LANGUAGE_DICT.get(st.session_state.tts_language, "en")
    # audio_data = text_to_speech_conversion(response_text, lang_code, engine=st.session_state.tts_engine)
    # if audio_data: st.audio(audio_data, format='audio/mp3', autoplay=True)
