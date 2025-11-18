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
# NOTE: For TTS to work, you must install the required libraries:
# For gTTS (generally easier): pip install gtts
# For edge_tts: pip install edge-tts
try:
    import edge_tts
except ImportError:
    edge_tts = None
    # st.info("Edge TTS is not available. Install 'edge-tts' for this option.")
try:
    from gtts import gTTS
except ImportError:
    gTTS = None
    # st.info("gTTS is not available. Install 'gtts' for this option.")

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

# --- MASSIVELY EXPANDED KNOWLEDGE BASE (STATIC COURSE CONTENT) ---
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
def load_data(knowledge_base_text):
    """
    Loads data for course embeddings. Uses the static knowledge_base_text 
    for stability, ensuring the app works without a local 'courses.csv' file.
    """
    try:
        # Use StringIO to read the static CSV string as if it were a file
        courses_df = pd.read_csv(io.StringIO(knowledge_base_text))
    except Exception as e:
        st.error(f"Failed to process knowledge base text: {e}")
        return pd.DataFrame(), np.array([])
        
    # Data Cleaning and Preparation
    courses_df = courses_df.dropna(subset=['Title']).reset_index(drop=True)
    courses_df.columns = courses_df.columns.str.strip().str.lower()
    if 'skill tags' in courses_df.columns:
        courses_df = courses_df.rename(columns={'skill tags': 'skill_tags'})
    courses_df = courses_df.drop_duplicates()
    
    model = load_model()
    # Create search text for vector embedding
    courses_df['search_text'] = (
        courses_df['title'] + " " + courses_df['skill_tags'] + " " +
        courses_df['provider'] + " " + courses_df['level'] + " " +
        courses_df['prerequisites'] + " " + courses_df['duration']
    ).fillna('')
    
    course_embeddings = model.encode(courses_df['search_text'].tolist(), show_progress_bar=False)
    
    return courses_df, course_embeddings

# --- CORE RAG & RECOMMENDATION LOGIC ---

def generate_user_embedding(user_profile, model):
    """
    Prioritizes the Target Career Domain (Goal) for better recommendation matching.
    """
    goal = user_profile['target_domain']
    
    profile_text = (
        f"Goal: {goal}. Career focus is strictly on {goal}. "
        f"Seeking courses in {goal} and {goal}. "
        f"Education: {user_profile['education_level']} in {user_profile['major']}. "
        f"Existing Skills: {user_profile['technical_skills']}."
    )
    
    return model.encode([profile_text])[0].reshape(1, -1)

# Helper functions (map_prerequisite_level, map_course_level, generate_llm_rationale)
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
    
    ranked_courses['rationale'] = ranked_courses.apply(
        lambda row: generate_llm_rationale(llm_client, user_profile, row, row['timeline']), axis=1
    )
    return ranked_courses

def get_rag_context(query, courses_df, course_embeddings, model, top_k=5):
    """Retrieves the most relevant course data using vector search."""
    if courses_df.empty or course_embeddings.size == 0:
        return "Course data is unavailable."

    query_embed = model.encode([query])[0].reshape(1, -1)
    similarity_scores = cosine_similarity(query_embed, course_embeddings)[0]
    top_indices = np.argsort(similarity_scores)[::-1][:top_k]
    context = ""
    for i in top_indices:
        row = courses_df.iloc[i]
        context += (
            f"Course Title: {row['title']}, Provider: {row['provider']}, "
            f"Level: {row['level']}, Duration: {row['duration']}, "
            f"Skills: {row['skill_tags']}, Prerequisites: {row['prerequisites']}, "
            f"Link: {row['link']} \n"
        )
    return context.strip()

def run_rag_query(query, courses_df, course_embeddings, model, llm_client, static_kb_text):
    """
    RAG Query function that explicitly models the agent as having two tools.
    """
    if not llm_client:
        return "The AI Agent is not initialized. Please ensure the Gemini API key is set."
    
    # 1. Dynamic Course Retrieval (Vector Search)
    course_context_vector = get_rag_context(query, courses_df, course_embeddings, model)
    
    # 2. Combine all knowledge sources for the LLM
    # The LLM is instructed to use the most relevant "tool" (context or general knowledge)
    full_context = f"""
    --- TOOL 1: Course Catalog (Retrieved Document Context via Vector Search) ---
    {course_context_vector}
    
    --- TOOL 2: General LLM Knowledge (Static KB for Cross-reference) ---
    {static_kb_text}
    """
    
    rag_prompt = f"""
    You are the **PersonalAI Course Consultant** chatbot. You have access to two tools: the **Course Catalog** (provided below as context) and your **General Knowledge**.
    
    * **If the query is about a specific course, price, duration, link, or prerequisite, use TOOL 1 (the Course Catalog).** Summarize the details (title, provider, level, link) based ONLY on the catalog content. If the information is not in the catalog, state that.
    * **If the query is for a general definition, concept explanation (e.g., 'What is Docker?'), or soft skill advice, use TOOL 2 (your General Knowledge).** Do not use the Course Catalog for these general questions.

    User Query: "{query}"
    Context (TOOL 1): {full_context}
    """
    try:
        response = llm_client.models.generate_content(
            model='gemini-2.5-flash', contents=rag_prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"Error communicating with the Gemini model: {e}"

# --- TEXT-TO-SPEECH (TTS) LOGIC ---

def text_to_speech_conversion(text, lang_code, engine="gtts", lang_name="English"):
    # Truncate text for TTS to avoid hitting limits or timeout issues with external services
    if len(text) > 500:
        text = text[:500] + "..." 
        
    try:
        if not text.strip():
            raise ValueError("Text to convert is empty.")
            
        if engine == "edge_tts" and edge_tts is not None:
            # NOTE: Running async code via asyncio.run() can sometimes cause issues in Streamlit.
            voice_name = EDGE_TTS_VOICE_DICT.get(lang_name, "en-US-AriaNeural")
            communicate = edge_tts.Communicate(text, voice_name)
            audio_bytes = b""
            
            async def run_tts():
                nonlocal audio_bytes
                try:
                    async for chunk in communicate.stream():
                        if chunk["type"] == "audio":
                            if chunk["content"]:
                                audio_bytes += chunk["content"]
                        elif chunk["type"] == "error":
                            error_msg = chunk.get('content', 'Unknown Edge TTS service error.')
                            raise RuntimeError(f"Edge TTS service error: {error_msg}")
                except Exception as e:
                    raise e

            asyncio.run(run_tts())
            
            if not audio_bytes:
                raise RuntimeError("Edge TTS returned no audio data.")

            return io.BytesIO(audio_bytes)
            
        elif engine == "gtts" and gTTS is not None:
            tts = gTTS(text=text, lang=lang_code)
            mp3_fp = io.BytesIO()
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            return mp3_fp
            
        else:
            st.warning(f"TTS engine '{engine}' is selected but not functional or installed.")
            return None
            
    except Exception as e:
        st.warning(f"TTS Error: Could not generate speech with {engine}. Details: {e}")
        return None

# --- STREAMLIT UI CODE ---
# Initialization of session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "tts_enabled" not in st.session_state:
    st.session_state.tts_enabled = False
if "tts_language" not in st.session_state:
    st.session_state.tts_language = DEFAULT_LANGUAGE
# Determine the initial default engine based on availability
initial_tts_engine = "gtts" if gTTS else ("edge_tts" if edge_tts else "None")
if "tts_engine" not in st.session_state:
    st.session_state.tts_engine = initial_tts_engine


st.set_page_config(layout="wide", page_title="AI Learning Path Recommender")
st.title("💡 AI-Powered Personalized Learning Path Recommender")

# --- Initialize variables before the try block ---
COURSES_DF = pd.DataFrame() # Initialize to empty DataFrame
COURSE_EMBEDDINGS = np.array([]) # Initialize to empty array
MODEL = None
LLM_CLIENT = None

# Load data and models
try:
    # UPDATED: Pass the KNOWLEDGE_BASE_TEXT to load_data for stability
    COURSES_DF, COURSE_EMBEDDINGS = load_data(KNOWLEDGE_BASE_TEXT)
    MODEL = load_model()
    LLM_CLIENT = setup_llm()
    # Now that the loading succeeded, display the toast
    if not COURSES_DF.empty:
        st.toast(f"Knowledge Base loaded with {len(COURSES_DF)} unique courses.", icon="🧠")
    else:
        st.error("No course data was loaded from the knowledge base.")

except Exception as e:
    st.error(f"⚠️ Could not initialize system components. Check API keys and data files. Error: {e}.")
    
# Load Sample Profiles
try:
    # NOTE: The user has not provided 'profiles.json'. Using an empty dict for stability.
    # To fix this, create a 'profiles.json' file in your directory.
    SAMPLE_PROFILES = {} 
    # with open('profiles.json', 'r') as f:
    #     SAMPLE_PROFILES = json.load(f)
except Exception:
    SAMPLE_PROFILES = {}

col_input, col_output = st.columns([1, 2.5])

with col_input:
    st.header("👤 User Profile Input")
    profile_keys = ["Manual Input"] + list(SAMPLE_PROFILES.keys())
    profile_selection = st.selectbox("Load Sample Profile:", profile_keys)
    loaded_profile = SAMPLE_PROFILES[profile_selection] if profile_selection != "Manual Input" else {}
    st.subheader("Required Background")
    education_options = ["Bachelor's", "Master's", "PhD", "High School/GED", "Certificate"]
    
    # Fix for previous SyntaxError: Reformatted the st.selectbox call
    education_level = st.selectbox(
        "Education Level:", 
        education_options, 
        index=education_options.index(
            loaded_profile.get('education_level', "Bachelor's")
        )
    )
    
    major = st.text_input("Major/Degree:", value=loaded_profile.get('major', "Computer Science"))
    technical_skills = st.text_area("Technical Skills (comma separated):", value=loaded_profile.get('technical_skills', "Python, SQL, Data Analysis, Excel, Git"))
    soft_skills = st.text_area("Soft Skills (comma separated):", value=loaded_profile.get('soft_skills', "Communication, Leadership, Problem-Solving"))
    st.subheader("Goals & Preferences (Optional)")
    target_domain = st.text_input("Target Career Domain (e.g., Data Science, **SAP MM**, DevOps):", value=loaded_profile.get('target_domain', "Data Science"))
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
    
    # Execution logic for generating the path 
    if st.button("🚀 Generate Learning Path", type="primary"):
        if COURSES_DF.empty or LLM_CLIENT is None:
            st.warning("Cannot generate path: Core system components (data or AI agent) failed to load. Check errors above.")
        elif not USER_PROFILE['technical_skills'].strip() or not USER_PROFILE['target_domain'].strip():
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
            st.session_state['recommendations_df'] = recommendations_df
            st.session_state['path_generated'] = True 
            st.session_state.messages = [] 

    st.subheader("🗣️ PersonalAI Chat Settings")
    st.session_state.tts_enabled = st.checkbox("Enable Text-to-Speech (TTS) Reply", value=st.session_state.tts_enabled)
    if st.session_state.tts_enabled:
        selected_lang_name = st.selectbox("Select Voice Language:", list(LANGUAGE_DICT.keys()), key='tts_language_selector', index=list(LANGUAGE_DICT.keys()).index(st.session_state.tts_language) if st.session_state.tts_language in LANGUAGE_DICT else 0)
        st.session_state.tts_language = selected_lang_name
        
        available_engines = []
        if gTTS: available_engines.append("gtts")
        if edge_tts: available_engines.append("edge_tts")
        if not available_engines: available_engines.append("None (Install gtts or edge-tts)")

        # Ensure the current engine is still valid or select the best available
        if st.session_state.tts_engine not in available_engines or st.session_state.tts_engine == "None":
            st.session_state.tts_engine = available_engines[0] if available_engines and available_engines[0] != "None (Install gtts or edge-tts)" else "None"
            
        current_engine_index = available_engines.index(st.session_state.tts_engine) if st.session_state.tts_engine in available_engines else 0
        st.session_state.tts_engine = st.selectbox("TTS Engine", available_engines, index=current_engine_index)

with col_output:
    st.markdown("## 🧠 Recommendation and Chat Output")
    
    # Display logic for recommendations
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

    # --- RAG CHATBOT UI ---
    st.divider()
    st.header("💬 PersonalAI Course Recommender (RAG Agent)")
    st.caption("Ask questions about the courses in the catalog (e.g., 'What is the deep learning specialization?' or 'Find the Docker course link').")
    
    if COURSES_DF.empty or LLM_CLIENT is None:
        st.warning("Chat is disabled: Data or AI Agent failed to load during initialization.")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    if not COURSES_DF.empty and LLM_CLIENT is not None:
        if prompt := st.chat_input("Ask a question about the courses in the catalog or a general tech concept..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Searching catalog and knowledge base..."):
                    # Call RAG with the static KB text included
                    response_text = run_rag_query(prompt, COURSES_DF, COURSE_EMBEDDINGS, MODEL, LLM_CLIENT, KNOWLEDGE_BASE_TEXT)
                st.markdown(response_text)
                
                # --- TTS EXECUTION ---
                if st.session_state.tts_enabled and st.session_state.tts_engine != "None":
                    lang_name = st.session_state.tts_language
                    lang_code = LANGUAGE_DICT.get(lang_name, "en")
                    tts_engine = st.session_state.tts_engine
                    
                    audio_data = text_to_speech_conversion(
                        response_text, lang_code, engine=tts_engine, lang_name=lang_name
                    )
                    
                    if audio_data:
                        st.audio(audio_data, format="audio/mp3", autoplay=True)
                # --- END TTS EXECUTION ---
                
                st.session_state.messages.append({"role": "assistant", "content": response_text})
