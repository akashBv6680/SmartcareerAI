import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import io
import asyncio # Added for edge_tts compatibility
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
EDGE_TTS_VOICE_DICT = {
    "English": "en-US-AriaNeural",
    "Spanish": "es-ES-ElviraNeural",
    "Arabic": "ar-EG-SalmaNeural",
    "French": "fr-FR-DeniseNeural",
    "German": "de-DE-KatjaNeural",
    "Hindi": "hi-IN-SwaraNeural",
    "Tamil": "ta-IN-PallaviNeural",
    "Bengali": "bn-IN-TanishaaNeural",
    "Japanese": "ja-JP-NanamiNeural",
    "Korean": "ko-KR-SunHiNeural",
    "Russian": "ru-RU-SvetlanaNeural",
    "Chinese (Simplified)": "zh-CN-XiaoxiaoNeural",
    "Portuguese": "pt-PT-FernandaNeural",
    "Italian": "it-IT-ElsaNeural",
    "Dutch": "nl-NL-ColetteNeural",
    "Turkish": "tr-TR-EmelNeural"
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
    except Exception as e:
        st.error(f"Error loading 'courses.csv': {e}")
        st.stop()
        
    courses_df.columns = courses_df.columns.str.strip().str.lower()
    if 'skill tags' in courses_df.columns:
        courses_df = courses_df.rename(columns={'skill tags': 'skill_tags'})
    model = load_model()
    # Including all text fields for embedding relevance calculation
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

def map_prerequisite_level(level_str):
    mapping = {'none': 0, 'basic': 1, 'beginner': 1, 'intermediate': 2, 'advanced': 3}
    if pd.isna(level_str):
        return 0
    cleaned_level = str(level_str).strip().lower()
    return mapping.get(cleaned_level, 0)

def map_course_level(level_str):
    mapping = {'beginner': 1, 'intermediate': 2, 'advanced': 3}
    if pd.isna(level_str):
        return 0
    cleaned_level = str(level_str).strip().lower()
    if '/' in cleaned_level:
        cleaned_level = cleaned_level.split('/')[0]
    return mapping.get(cleaned_level, 0)

def generate_llm_rationale(client, user_profile, course_row, timeline_type):
    if not client:
        return f"LLM Rationale Unavailable. Heuristic: Good fit for {course_row['level']} level. It is a {timeline_type} step."
    prompt = f"""
    Act as a highly experienced Career Counselor. Given the user profile and the recommended course,
    provide a **concise, two-sentence rationale** (less than 40 words total).

    ---
    User Profile:
    - Education: {user_profile['education_level']} in {user_profile['major']}
    - Skills: {user_profile['technical_skills']}
    - Goal: Career switch/growth into {user_profile['target_domain']}

    Course Recommended:
    - Title: {course_row['title']} by {course_row['provider']}
    - Level: {course_row['level']} ({timeline_type} plan)
    - Key Skills Taught: {course_row['skill_tags']}
    - Prerequisites: {course_row['prerequisites']}
    ---

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

    # Level and prerequisite logic
    user_level = 1 # Beginner
    if 'intermediate' in user_profile['technical_skills'].lower() or user_profile['education_level'] in ['Master\'s', 'PhD']:
        user_level = 2
    if 'advanced' in user_profile['technical_skills'].lower() or user_profile['education_level'] == 'PhD':
        user_level = 3

    results_df['course_level_num'] = results_df['level'].apply(map_course_level)
    results_df['prereq_level_num'] = results_df['prerequisites'].apply(
        lambda x: 0 if pd.isna(x) else map_prerequisite_level(str(x).split(',')[0].strip())
    )
    results_df['prereq_penalty'] = np.where(
        results_df['prereq_level_num'] > user_level, 0.5, 1.0
    )
    results_df['fit_score'] = (results_df['similarity_score'] * 100 * results_df['prereq_penalty']).round(1)
    ranked_courses = results_df.sort_values(by='fit_score', ascending=False).head(10).copy()

    def assign_timeline(row):
        is_basic = row['level'] in ['Beginner', 'Intermediate']
        duration_lower = str(row['duration']).lower()
        is_short = 'week' in duration_lower or ('month' in duration_lower and int(duration_lower.split()[0]) <= 2)
        if is_basic and is_short and row['fit_score'] >= 50:
            return 'Short-Term'
        elif row['fit_score'] >= 40:
            return 'Long-Term'
        return 'Long-Term'
    ranked_courses['timeline'] = ranked_courses.apply(assign_timeline, axis=1)
    
    # This is the heavy LLM call, which must only be run once.
    ranked_courses['rationale'] = ranked_courses.apply(
        lambda row: generate_llm_rationale(llm_client, user_profile, row, row['timeline']), axis=1
    )
    return ranked_courses

def get_rag_context(query, courses_df, course_embeddings, model, top_k=5):
    """Retrieves the most relevant course data using vector search, including the link."""
    query_embed = model.encode([query])[0].reshape(1, -1)
    similarity_scores = cosine_similarity(query_embed, course_embeddings)[0]
    top_indices = np.argsort(similarity_scores)[::-1][:top_k]
    context = ""
    for i in top_indices:
        row = courses_df.iloc[i]
        # UPDATED: The code already had the full fields, including 'link'.
        context += (
            f"Course Title: {row['title']}, Provider: {row['provider']}, "
            f"Level: {row['level']}, Duration: {row['duration']}, "
            f"Skills: {row['skill_tags']}, Prerequisites: {row['prerequisites']}, "
            f"Link: {row['link']} \n"
        )
    return context.strip()

def run_rag_query(query, courses_df, course_embeddings, model, llm_client):
    if not llm_client:
        return "The AI Agent is not initialized."
    context = get_rag_context(query, courses_df, course_embeddings, model)
    rag_prompt = f"""
    You are the **PersonalAI Course Recommender** chatbot. Your goal is to answer questions about learning paths and courses based *only* on the provided context. If the user asks for a link, provide the URL found in the context.

    User Query: "{query}"
    Context (Relevant Courses): ---
    {context}
    ---
    Provide a concise, helpful answer. If context does not contain the answer, state that you cannot find the information in the current catalog.
    """
    try:
        response = llm_client.models.generate_content(
            model='gemini-2.5-flash', contents=rag_prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"Error communicating with the Gemini model: {e}"

def text_to_speech_conversion(text, lang_code, engine="gtts", lang_name="English"):
    try:
        if engine == "edge_tts" and edge_tts is not None:
            voice_name = EDGE_TTS_VOICE_DICT.get(lang_name, "en-US-AriaNeural")
            communicate = edge_tts.Communicate(text, voice_name)
            audio_bytes = b""
            # Stream the audio to get all chunks
            async def run_tts():
                nonlocal audio_bytes
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_bytes += chunk["content"]
            asyncio.run(run_tts())
            return io.BytesIO(audio_bytes)
        elif engine == "gtts" and gTTS is not None:
            tts = gTTS(text=text, lang=lang_code)
            mp3_fp = io.BytesIO()
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            return mp3_fp
        else:
            st.warning("No functional TTS engine available for audio playback.")
            return None
    except Exception as e:
        st.warning(f"TTS Error: Could not generate speech. Details: {e}")
        return None

# --- STREAMLIT UI CODE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "tts_enabled" not in st.session_state:
    st.session_state.tts_enabled = False
if "tts_language" not in st.session_state:
    st.session_state.tts_language = DEFAULT_LANGUAGE
if "tts_engine" not in st.session_state:
    st.session_state.tts_engine = "gtts" if gTTS else ("edge_tts" if edge_tts else "None")
# FIX: Initialize recommendations DataFrame to avoid re-running expensive LLM calls
if "recommendations_df" not in st.session_state:
    st.session_state.recommendations_df = pd.DataFrame()


st.set_page_config(layout="wide", page_title="AI Learning Path Recommender")
st.title("💡 AI-Powered Personalized Learning Path Recommender")
try:
    COURSES_DF, COURSE_EMBEDDINGS = load_data()
    MODEL = load_model()
    LLM_CLIENT = setup_llm()
except Exception as e:
    st.error(f"Could not initialize system components: {e}.")

# FIX: Broaden try-except block to catch JSONDecodeError, which was likely the silent error
try:
    with open('profiles.json', 'r') as f:
        SAMPLE_PROFILES = json.load(f)
except FileNotFoundError:
    SAMPLE_PROFILES = {}
except json.JSONDecodeError as e:
    st.error(f"Error: 'profiles.json' file is corrupted (Invalid JSON format). Please fix the file. Details: {e}")
    SAMPLE_PROFILES = {}
except Exception as e:
    st.warning(f"Warning: Could not load sample profiles. {e}")
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
    
    # FIX: Calculate and save results to session state ONLY on button click
    if st.button("🚀 Generate Learning Path", type="primary"):
        if not USER_PROFILE['technical_skills'].strip() or not USER_PROFILE['target_domain'].strip():
            st.error("Please provide at least your Technical Skills and Target Career Domain.")
        else:
            with st.spinner("Analyzing profile, computing similarity, and generating LLM rationale..."):
                recommendations_df = recommend_courses(
                    USER_PROFILE,
                    COURSES_DF,
                    COURSE_EMBEDDINGS,
                    MODEL,
                    LLM_CLIENT
                )
            # Store results and set flag
            st.session_state['recommendations_df'] = recommendations_df
            st.session_state['path_generated'] = True 
            st.session_state.messages = [] # Clear chat history on new path generation

    st.subheader("🗣️ PersonalAI Chat Settings")
    st.session_state.tts_enabled = st.checkbox("Enable Text-to-Speech (TTS) Reply", value=st.session_state.tts_enabled)
    if st.session_state.tts_enabled:
        selected_lang_name = st.selectbox("Select Voice Language:", list(LANGUAGE_DICT.keys()), key='tts_language_selector', index=list(LANGUAGE_DICT.keys()).index(st.session_state.tts_language) if st.session_state.tts_language in LANGUAGE_DICT else 0)
        st.session_state.tts_language = selected_lang_name
        
        available_engines = []
        if gTTS: available_engines.append("gtts")
        if edge_tts: available_engines.append("edge_tts")
        if not available_engines: available_engines.append("None")

        current_engine_index = available_engines.index(st.session_state.tts_engine) if st.session_state.tts_engine in available_engines else 0
        st.session_state.tts_engine = st.selectbox("TTS Engine", available_engines, index=current_engine_index)

with col_output:
    st.markdown("## 🧠 Recommendation and Chat Output")
    
    # FIX: Check if the path has been generated and use the stored DataFrame
    if st.session_state.get('path_generated', False) and not st.session_state.recommendations_df.empty:
        recommendations_df = st.session_state.recommendations_df
        target_domain = USER_PROFILE['target_domain']

        st.markdown(f"### 🎯 Learning Path for **{target_domain}**")
        st.markdown(f"**Based on:** {USER_PROFILE['technical_skills']}")
        
        # --- SHORT-TERM PLAN ---
        st.divider()
        st.subheader("🗓️ Short-Term Plan (Next 1-3 Months)")
        st.caption("Foundational, high-impact courses for immediate skill gain.")
        short_term = recommendations_df[recommendations_df['timeline'] == 'Short-Term']
        if not short_term.empty:
            for i, row in short_term.iterrows():
                st.success(f"**{row['title']}** ({row['provider']})")
                cols = st.columns([1, 1, 1, 4])
                cols[0].metric("Fit Score", f"{row['fit_score']}%")
                cols[1].metric("Level", row['level'])
                cols[2].metric("Duration", row['duration'])
                cols[3].markdown(f"**Rationale:** {row['rationale']}")
                st.markdown(f"**Enroll:** [Access Course Link Here]({row['link']})")
                st.markdown("---")
        else:
            st.info("No courses prioritized for the short term based on current criteria.")
        
        # --- LONG-TERM PLAN ---
        st.divider()
        st.subheader("📚 Long-Term Plan (Next 3-12 Months)")
        st.caption("Specialization and advanced certifications to achieve your career goal.")
        long_term = recommendations_df[recommendations_df['timeline'] == 'Long-Term']
        if not long_term.empty:
            for i, row in long_term.iterrows():
                st.info(f"**{row['title']}** ({row['provider']})")
                cols = st.columns([1, 1, 1, 4])
                cols[0].metric("Fit Score", f"{row['fit_score']}%")
                cols[1].metric("Level", row['level'])
                cols[2].metric("Duration", row['duration'])
                cols[3].markdown(f"**Rationale:** {row['rationale']}")
                st.markdown(f"**Enroll:** [Access Course Link Here]({row['link']})")
                st.markdown("---")
        else:
            st.info("No courses recommended for the long term.")
    else:
        st.info("Please set up your profile on the left and click 'Generate Learning Path' to view recommendations.")

    # --- RAG CHATBOT UI ---
    st.divider()
    st.header("💬 PersonalAI Course Recommender (RAG Agent)")
    st.caption("Ask questions about the courses in the catalog (e.g., 'What are the prerequisites for the AWS course?' or 'Tell me about the Data Science beginner courses').")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if prompt := st.chat_input("Ask a question about the courses in the catalog..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Searching catalog and thinking..."):
                response_text = run_rag_query(prompt, COURSES_DF, COURSE_EMBEDDINGS, MODEL, LLM_CLIENT)
            st.markdown(response_text)
            if st.session_state.tts_enabled and st.session_state.tts_engine != "None":
                lang_name = st.session_state.tts_language
                lang_code = LANGUAGE_DICT.get(lang_name, "en")
                tts_engine = st.session_state.tts_engine
                
                audio_data = text_to_speech_conversion(
                    response_text, lang_code, engine=tts_engine, lang_name=lang_name
                )
                
                if audio_data:
                    st.audio(audio_data, format="audio/mp3", autoplay=True)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
