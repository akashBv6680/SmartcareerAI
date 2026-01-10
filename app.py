import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import io
import asyncio
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai

# --- TTS ENGINE SETUP ---
try:
    import edge_tts
except ImportError:
    edge_tts = None
try:
    from gtts import gTTS
except ImportError:
    gTTS = None

# --- CONFIGURATION ---
LANGUAGE_DICT = {
    "English": "en", "Spanish": "es", "Arabic": "ar", "French": "fr", "German": "de",
    "Hindi": "hi", "Tamil": "ta", "Bengali": "bn", "Japanese": "ja", "Korean": "ko",
    "Russian": "ru", "Chinese (Simplified)": "zh-cn", "Portuguese": "pt", "Italian": "it",
    "Dutch": "nl", "Turkish": "tr"
}
EDGE_TTS_VOICE_DICT = {
    "English": "en-US-AriaNeural", "Spanish": "es-ES-ElviraNeural", "Arabic": "ar-EG-SalmaNeural",
    "French": "fr-FR-DeniseNeural", "German": "de-DE-KatjaNeural", "Hindi": "hi-IN-SwaraNeural",
    "Tamil": "ta-IN-PallaviNeural", "Bengali": "bn-IN-TanishaaNeural", "Japanese": "ja-JP-NanamiNeural",
    "Korean": "ko-KR-SunHiNeural", "Russian": "ru-RU-SvetlanaNeural", "Chinese (Simplified)": "zh-CN-XiaoxiaoNeural",
    "Portuguese": "pt-PT-FernandaNeural", "Italian": "it-IT-ElsaNeural", "Dutch": "nl-NL-ColetteNeural",
    "Turkish": "tr-TR-EmelNeural"
}
DEFAULT_LANGUAGE = "English"

# --- MASSIVELY EXPANDED KNOWLEDGE BASE ---
KNOWLEDGE_BASE_TEXT = """
Title,Provider,Duration,Prerequisites,Skill Tags,Level,Link
Python Crash Course,Coursera (Google),4 Weeks,None,"Python, Basics, Programming, Data Types",Beginner,https://www.coursera.org/learn/python-crash-course
Data Science Fundamentals,edX,6 Weeks,"Basic Math, Algebra","Data Science, Statistics, R, Visualization",Beginner,https://www.edx.org/learn/data-science
Machine Learning Specialization,Coursera (Stanford/DeepLearning.AI),5 Months,"Calculus, Python, Linear Algebra","Machine Learning, Algorithms, Neural Networks, Deep Learning",Advanced,https://www.coursera.org/specializations/machine-learning-introduction
Agile Project Management,PMI,8 Weeks,None,"Agile, Scrum, Project Management, Communication",Intermediate,https://www.pmi.org/certifications/agile-acp
Cloud Computing with AWS,AWS Training,10 Weeks,Basic Networking,"AWS, Cloud, Infrastructure, Networking, DevOps",Intermediate,https://aws.amazon.com/training/
Effective Communication,LinkedIn Learning,2 Weeks,None,"Communication, Presentation, Leadership, Soft Skills",Beginner,https://www.linkedin.com/learning/topics/communication
Advanced SQL and Database Design,Udemy,6 Weeks,Basic SQL,"SQL, Database, Normalization, Query Optimization",Intermediate,https://www.udemy.com/topic/sql/
Introduction to Web Development,FreeCodeCamp,3 Months,None,"HTML, CSS, JavaScript, Web Dev",Beginner,https://www.freecodecamp.org/learn
TensorFlow Developer Certificate,DeepLearning.AI (Google),4 Months,"Advanced Python, ML Fundamentals","TensorFlow, Deep Learning, Neural Networks, Computer Vision",Advanced,https://www.deeplearning.ai/courses/tensorflow-developer-professional-certificate/
SAP MM Certification Course,Finprov Learning,2 Months,None,"SAP MM, Materials Management, Procurement, Inventory",Beginner/Intermediate,https://finprov.com/course/sap-mm-certification-course/
SAP MM Online Training,Besant Technologies,1 Month,None,"SAP MM, Purchase Order, Invoice Verification, Stock Management",Beginner/Intermediate,https://www.besanttechnologies.com/training-courses/sap-courses/sap-mm-training
SAP Professional Fundamentals,Coursera,4 Weeks,None,"SAP Ecosystem, Business Processes, ERP Fundamentals",Beginner,https://www.coursera.org/learn/sap-professional-fundamentals
Introduction to SAP,Alison,4-5 Hours,None,"SAP Navigation, Core Transactions, ERP Basics",Beginner,https://alison.com/course/an-introduction-to-sap
Data Engineering on Google Cloud,Google Cloud Training,3 Months,"Python, SQL","BigQuery, Data Pipelines, Apache Beam, Cloud Storage",Intermediate,https://cloud.google.com/training/data-engineering-and-analytics
Full Stack Web Development,Udacity,4 Months,Basic HTML/CSS,"React, Node.js, APIs, Databases, Frontend/Backend",Intermediate,https://www.udacity.com/course/full-stack-web-developer-nanodegree--nd0044
Power BI Data Analytics,Microsoft Learn,6 Weeks,Basic Excel,"Power BI, DAX, Data Visualization, Business Intelligence",Beginner,https://learn.microsoft.com/en-us/training/powerplatform/power-bi
Tableau Desktop Specialist,Tableau/Salesforce,5 Weeks,None,"Tableau, Data Visualization, Dashboard Design, Analytics",Beginner,https://www.tableau.com/learn/training
AI For Everyone,Coursera (DeepLearning.AI),4 Weeks,None,"AI Strategy, Machine Learning Basics, AI Applications",Beginner,https://www.coursera.org/learn/ai-for-everyone
Generative AI with LLMs,Coursera (DeepLearning.AI),3 Months,"Python, ML Basics","LLMs, Prompt Engineering, GPT, Generative AI, Fine-tuning",Intermediate,https://www.coursera.org/learn/generative-ai-with-llms
Docker and Kubernetes,Udemy,8 Weeks,Basic Linux,"Docker, Kubernetes, Containers, DevOps, Microservices",Intermediate,https://www.udemy.com/topic/docker/
Cybersecurity Fundamentals,Coursera (IBM),3 Months,None,"Cybersecurity, Network Security, Threat Analysis, Encryption",Beginner,https://www.coursera.org/professional-certificates/ibm-cybersecurity-analyst
Azure Data Fundamentals,Microsoft Learn,6 Weeks,None,"Azure, Data Storage, SQL, NoSQL, Data Analytics",Beginner,https://learn.microsoft.com/en-us/certifications/azure-data-fundamentals/
Java Programming Masterclass,Udemy,12 Weeks,None,"Java, OOP, Spring Boot, Multithreading, Data Structures",Beginner,https://www.udemy.com/course/java-the-complete-java-developer-course/
Natural Language Processing,Coursera (DeepLearning.AI),4 Months,"Python, Deep Learning","NLP, Transformers, BERT, Text Processing, Sentiment Analysis",Advanced,https://www.coursera.org/specializations/natural-language-processing
Blockchain Basics,Coursera (University at Buffalo),4 Weeks,None,"Blockchain, Cryptocurrency, Smart Contracts, Distributed Systems",Beginner,https://www.coursera.org/learn/blockchain-basics
PySpark for Big Data,Udemy,6 Weeks,"Python, Spark Basics","PySpark, Big Data, Spark SQL, Data Processing, ETL",Intermediate,https://www.udemy.com/topic/apache-spark/
Git and GitHub Essentials,Udemy,3 Weeks,None,"Git, GitHub, Version Control, Collaboration, CI/CD",Beginner,https://www.udemy.com/topic/git/
MLOps Specialization,Coursera (DeepLearning.AI),4 Months,"ML Fundamentals, Python","MLOps, Model Deployment, CI/CD, Model Monitoring",Advanced,https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops
Excel to Power Query and Power Pivot,Udemy,5 Weeks,Basic Excel,"Excel, Power Query, Power Pivot, Data Modeling",Intermediate,https://www.udemy.com/topic/microsoft-power-query/
SAP FICO Training,SLA Consultants,2 Months,None,"SAP FICO, Financial Accounting, Controlling, SAP ERP",Beginner/Intermediate,https://www.slaconsultantsindia.com/sap-fico-training
Deep Learning Specialization,Coursera (DeepLearning.AI),5 Months,"Python, ML Basics","Deep Learning, CNNs, RNNs, Neural Networks, Hyperparameter Tuning",Advanced,https://www.coursera.org/specializations/deep-learning
React - The Complete Guide,Udemy,10 Weeks,JavaScript Basics,"React, Redux, Hooks, Context API, Frontend Development",Intermediate,https://www.udemy.com/course/react-the-complete-guide-incl-redux/
Statistics for Data Science,Coursera (Stanford),6 Weeks,Basic Math,"Statistics, Probability, Hypothesis Testing, Regression",Beginner,https://www.coursera.org/learn/stanford-statistics
"""

# --- LLM SETUP AND DATA LOADING ---
def setup_llm():
    try:
        api_key = os.environ.get('GEMINI_API_KEY') or st.secrets.get('GEMINI_API_KEY')
        if not api_key:
            st.error("🔑 Error: GEMINI_API_KEY not found.")
            return None
        return genai.Client(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize Gemini Client: {e}")
        return None

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_data(knowledge_base_text):
    try:
        courses_df = pd.read_csv(io.StringIO(knowledge_base_text))
    except Exception as e:
        st.error(f"Failed to process knowledge base text: {e}")
        return pd.DataFrame(), np.array([])
    
    courses_df = courses_df.dropna(subset=['Title']).reset_index(drop=True)
    courses_df.columns = courses_df.columns.str.strip().str.lower()
    if 'skill tags' in courses_df.columns:
        courses_df = courses_df.rename(columns={'skill tags': 'skill_tags'})
    courses_df = courses_df.drop_duplicates()
    
    model = load_model()
    courses_df['search_text'] = (
        courses_df['title'] + " " + courses_df['skill_tags'] + " " +
        courses_df['provider'] + " " + courses_df['level'] + " " +
        courses_df['prerequisites'] + " " + courses_df['duration']
    ).fillna('')
    
    course_embeddings = model.encode(courses_df['search_text'].tolist(), show_progress_bar=False)
    return courses_df, course_embeddings

# --- CORE RAG & RECOMMENDATION LOGIC ---

def generate_user_embedding(user_profile, model):
    goal = user_profile['target_domain']
    profile_text = (
        f"Goal: {goal}. Career focus is strictly on {goal}. "
        f"Education: {user_profile['education_level']} in {user_profile['major']}. "
        f"Existing Skills: {user_profile['technical_skills']}."
    )
    return model.encode([profile_text])[0].reshape(1, -1)

def map_prerequisite_level(level_str):
    mapping = {'none': 0, 'basic': 1, 'beginner': 1, 'intermediate': 2, 'advanced': 3}
    if pd.isna(level_str): return 0
    return mapping.get(str(level_str).strip().lower(), 0)

def map_course_level(level_str):
    mapping = {'beginner': 1, 'intermediate': 2, 'advanced': 3}
    if pd.isna(level_str): return 0
    cleaned = str(level_str).strip().lower().split('/')[0]
    return mapping.get(cleaned, 0)

def generate_llm_rationale(client, user_profile, course_row, timeline_type):
    if not client: return "Rationale Unavailable."
    prompt = f"Explain why {course_row['title']} fits a user aiming for {user_profile['target_domain']} in 2 short sentences."
    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash', contents=prompt, config={"max_output_tokens": 100}
        )
        return response.text.strip()
    except: return "Course fits your profile level."

def recommend_courses(user_profile, courses_df, course_embeddings, model, llm_client):
    user_embed = generate_user_embedding(user_profile, model)
    similarity_scores = cosine_similarity(user_embed, course_embeddings)[0]
    results_df = courses_df.copy()
    results_df['similarity_score'] = similarity_scores
    
    user_level = 1
    if 'intermediate' in user_profile['technical_skills'].lower(): user_level = 2
    if 'advanced' in user_profile['technical_skills'].lower(): user_level = 3

    results_df['course_level_num'] = results_df['level'].apply(map_course_level)
    results_df['prereq_penalty'] = results_df['level'].apply(lambda x: 1.0 if map_course_level(x) <= user_level else 0.5)
    results_df['fit_score'] = (results_df['similarity_score'] * 100 * results_df['prereq_penalty']).round(1)
    
    ranked_courses = results_df.sort_values(by='fit_score', ascending=False).head(10).copy()
    ranked_courses['timeline'] = ranked_courses['level'].apply(lambda x: 'Short-Term' if 'Beginner' in x else 'Long-Term')
    ranked_courses['rationale'] = ranked_courses.apply(lambda row: generate_llm_rationale(llm_client, user_profile, row, row['timeline']), axis=1)
    return ranked_courses

def get_rag_context(query, courses_df, course_embeddings, model, top_k=3):
    query_embed = model.encode([query])[0].reshape(1, -1)
    similarity_scores = cosine_similarity(query_embed, course_embeddings)[0]
    top_indices = np.argsort(similarity_scores)[::-1][:top_k]
    context = ""
    for i in top_indices:
        row = courses_df.iloc[i]
        context += f"Course: {row['title']}, Provider: {row['provider']}, Link: {row['link']}\n"
    return context

def run_rag_query(query, courses_df, course_embeddings, model, llm_client, static_kb_text):
    if not llm_client: return "AI Agent is not initialized."
    target_language_name = st.session_state.get('tts_language', 'English')
    course_context = get_rag_context(query, courses_df, course_embeddings, model)
    
    # QUOTA PROTECTED PROMPT: Prevents 5+ minute voice replies
    rag_prompt = f"""
    You are the PersonalAI Consultant. Respond ONLY in {target_language_name}.
    KEEP YOUR RESPONSE CONCISE (under 150 words) to ensure voice stability.
    Answer the user query using the catalog if relevant. 
    
    USER QUERY: "{query}"
    CATALOG CONTEXT: {course_context}
    """
    try:
        response = llm_client.models.generate_content(
            model='gemini-2.0-flash', contents=rag_prompt, config={"max_output_tokens": 400}
        )
        return response.text.strip()
    except Exception as e: return f"Error: {e}"

# --- IMPROVED TTS LOGIC (The Fixed Version) ---
def text_to_speech_conversion(text, lang_code, engine="gtts", lang_name="English"):
    if len(text) > 2000: # Increased from 500
        text = text[:1997] + "..." 
        
    try:
        if engine == "edge_tts" and edge_tts is not None:
            voice_name = EDGE_TTS_VOICE_DICT.get(lang_name, "en-US-AriaNeural")
            communicate = edge_tts.Communicate(text, voice_name)
            
            async def run_tts():
                audio_bytes = b""
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio": audio_bytes += chunk["content"]
                return audio_bytes

            # Fix: Safe Async Loop for Streamlit
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            audio_data = loop.run_until_complete(run_tts())
            loop.close()
            return io.BytesIO(audio_data)
            
        elif engine == "gtts" and gTTS is not None:
            tts = gTTS(text=text, lang=lang_code)
            mp3_fp = io.BytesIO()
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            return mp3_fp
        return None
    except Exception as e:
        st.warning(f"TTS Error: {e}")
        return None

# --- STREAMLIT UI CODE ---
if "messages" not in st.session_state: st.session_state.messages = []
if "tts_enabled" not in st.session_state: st.session_state.tts_enabled = False
if "tts_language" not in st.session_state: st.session_state.tts_language = DEFAULT_LANGUAGE
if "last_audio" not in st.session_state: st.session_state.last_audio = None # Persistence Fix

st.set_page_config(layout="wide", page_title="AI Learning Path Recommender")
st.title("💡 AI-Powered Personalized Learning Path Recommender")

COURSES_DF, COURSE_EMBEDDINGS = load_data(KNOWLEDGE_BASE_TEXT)
MODEL = load_model()
LLM_CLIENT = setup_llm()

col_input, col_output = st.columns([1, 2.5])

with col_input:
    st.header("👤 User Profile Input")
    education_level = st.selectbox("Education Level:", ["Bachelor's", "Master's", "PhD", "High School/GED"])
    major = st.text_input("Major/Degree:", value="Computer Science")
    technical_skills = st.text_area("Technical Skills:", value="Python, SQL")
    target_domain = st.text_input("Target Career Domain:", value="Data Science")
    
    if st.button("🚀 Generate Learning Path", type="primary"):
        USER_PROFILE = {'education_level': education_level, 'major': major, 'technical_skills': technical_skills, 'target_domain': target_domain}
        st.session_state['recommendations_df'] = recommend_courses(USER_PROFILE, COURSES_DF, COURSE_EMBEDDINGS, MODEL, LLM_CLIENT)
        st.session_state['path_generated'] = True

    st.subheader("🗣️ Voice Settings")
    st.session_state.tts_enabled = st.checkbox("Enable Voice Reply", value=st.session_state.tts_enabled)
    st.session_state.tts_language = st.selectbox("Voice Language:", list(LANGUAGE_DICT.keys()))
    st.session_state.tts_engine = st.selectbox("TTS Engine:", ["gtts", "edge_tts"] if edge_tts and gTTS else ["gtts"])

with col_output:
    if st.session_state.get('path_generated', False):
        st.markdown(f"### 🎯 Path for {target_domain}")
        st.dataframe(st.session_state.recommendations_df[['title', 'provider', 'fit_score', 'timeline']])

    st.divider()
    st.header("💬 PersonalAI Chatbot")
    
    # Display message history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Generating..."):
                response_text = run_rag_query(prompt, COURSES_DF, COURSE_EMBEDDINGS, MODEL, LLM_CLIENT, KNOWLEDGE_BASE_TEXT)
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})

                if st.session_state.tts_enabled:
                    l_code = LANGUAGE_DICT.get(st.session_state.tts_language, "en")
                    audio_data = text_to_speech_conversion(response_text, l_code, st.session_state.tts_engine, st.session_state.tts_language)
                    if audio_data:
                        # Save to session state to prevent disappearance
                        st.session_state.last_audio = audio_data.getvalue()
                        st.audio(st.session_state.last_audio, format="audio/mp3", autoplay=True)

    # Persistence Player: Keeps the last audio available even if you scroll or click widgets
    if st.session_state.last_audio:
        with st.sidebar:
            st.info("🎵 Last Voice Reply")
            st.audio(st.session_state.last_audio, format="audio/mp3")
