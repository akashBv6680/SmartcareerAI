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
    "English": "en", "Spanish": "es", "Arabic": "ar", "French": "fr", "German": "de", 
    "Hindi": "hi", "Tamil": "ta", "bn": "Bengali", "Japanese": "ja", "Korean": "ko", 
    "Russian": "ru", "Chinese (Simplified)": "zh-Hans", "Portuguese": "pt", "Italian": "it", 
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
# This static text is now the full course catalog, as explicitly requested by the user.
KNOWLEDGE_BASE_TEXT = """
TitleProviderDurationPrerequisitesSkill TagsLevelLink
Python Crash CourseCoursera (Google)4 WeeksNonePython, Basics, Programming, Data TypesBeginnerhttps://www.coursera.org/learn/python-crash-course
Data Science FundamentalsedX6 WeeksBasic Math, AlgebraData Science, Statistics, R, VisualizationBeginnerhttps://www.edx.org/learn/data-science
Machine Learning SpecializationCoursera (Stanford/DeepLearning.AI)5 MonthsCalculus, Python, Linear AlgebraMachine Learning, Algorithms, Neural Networks, Deep LearningAdvancedhttps://www.coursera.org/specializations/machine-learning-introduction
Agile Project ManagementPMI8 WeeksNoneAgile, Scrum, Project Management, CommunicationIntermediatehttps://www.pmi.org/certifications/agile-acp
Cloud Computing with AWSAWS Training10 WeeksBasic NetworkingAWS, Cloud, Infrastructure, Networking, DevOpsIntermediatehttps://aws.amazon.com/training/
Effective CommunicationLinkedIn Learning2 WeeksNoneCommunication, Presentation, Leadership, Soft SkillsBeginnerhttps://www.linkedin.com/learning/topics/communication
Advanced SQL and Database DesignUdemy6 WeeksBasic SQLSQL, Database, Normalization, Query OptimizationIntermediatehttps://www.udemy.com/topic/sql/
Introduction to Web DevelopmentFreeCodeCamp3 MonthsNoneHTML, CSS, JavaScript, Web DevBeginnerhttps://www.freecodecamp.org/learn
TensorFlow Developer CertificateDeepLearning.AI (Google)4 MonthsAdvanced Python, ML FundamentalsTensorFlow, Deep Learning, Neural Networks, Computer VisionAdvancedhttps://www.deeplearning.ai/courses/tensorflow-developer-professional-certificate/
SAP MM Certification CourseFinprov Learning2 MonthsNoneSAP MM, Materials Management, Procurement, InventoryBeginner/Intermediatehttps://finprov.com/course/sap-mm-certification-course/
SAP MM Online TrainingBesant Technologies1 MonthNoneSAP MM, Purchase Order, Invoice Verification, Stock ManagementBeginner/Intermediatehttps://www.besanttechnologies.com/training-courses/sap-courses/sap-mm-training
SAP Professional FundamentalsCoursera4 WeeksNoneSAP Ecosystem, Business Processes, ERP FundamentalsBeginnerhttps://www.coursera.org/learn/sap-professional-fundamentals
Introduction to SAPAlison4-5 HoursNoneSAP Navigation, Core Transactions, ERP BasicsBeginnerhttps://alison.com/course/an-introduction-to-sap
Data Engineering on Google CloudGoogle Cloud Training3 MonthsPython, SQLBigQuery, Data Pipelines, Apache Beam, Cloud StorageIntermediatehttps://cloud.google.com/training/data-engineering-and-analytics
Full Stack Web DevelopmentUdacity4 MonthsBasic HTML/CSSReact, Node.js, APIs, Databases, Frontend/BackendIntermediatehttps://www.udacity.com/course/full-stack-web-developer-nanodegree--nd0044
Power BI Data AnalyticsMicrosoft Learn6 WeeksBasic ExcelPower BI, DAX, Data Visualization, Business IntelligenceBeginnerhttps://learn.microsoft.com/en-us/training/powerplatform/power-bi
Tableau Desktop SpecialistTableau/Salesforce5 WeeksNoneTableau, Data Visualization, Dashboard Design, AnalyticsBeginnerhttps://www.tableau.com/learn/training
AI For EveryoneCoursera (DeepLearning.AI)4 WeeksNoneAI Strategy, Machine Learning Basics, AI ApplicationsBeginnerhttps://www.coursera.org/learn/ai-for-everyone
Generative AI with LLMsCoursera (DeepLearning.AI)3 MonthsPython, ML BasicsLLMs, Prompt Engineering, GPT, Generative AI, Fine-tuningIntermediatehttps://www.coursera.org/learn/generative-ai-with-llms
Docker and KubernetesUdemy8 WeeksBasic LinuxDocker, Kubernetes, Containers, DevOps, MicroservicesIntermediatehttps://www.udemy.com/topic/docker/
Cybersecurity FundamentalsCoursera (IBM)3 MonthsNoneCybersecurity, Network Security, Threat Analysis, EncryptionBeginnerhttps://www.coursera.org/professional-certificates/ibm-cybersecurity-analyst
Azure Data FundamentalsMicrosoft Learn6 WeeksNoneAzure, Data Storage, SQL, NoSQL, Data AnalyticsBeginnerhttps://learn.microsoft.com/en-us/certifications/azure-data-fundamentals/
Java Programming MasterclassUdemy12 WeeksNoneJava, OOP, Spring Boot, Multithreading, Data StructuresBeginnerhttps://www.udemy.com/course/java-the-complete-java-developer-course/
Natural Language ProcessingCoursera (DeepLearning.AI)4 MonthsPython, Deep LearningNLP, Transformers, BERT, Text Processing, Sentiment AnalysisAdvancedhttps://www.coursera.org/specializations/natural-language-processing
Blockchain BasicsCoursera (University at Buffalo)4 WeeksNoneBlockchain, Cryptocurrency, Smart Contracts, Distributed SystemsBeginnerhttps://www.coursera.org/learn/blockchain-basics
PySpark for Big DataUdemy6 WeeksPython, Spark BasicsPySpark, Big Data, Spark SQL, Data Processing, ETLIntermediatehttps://www.udemy.com/topic/apache-spark/
Git and GitHub EssentialsUdemy3 WeeksNoneGit, GitHub, Version Control, Collaboration, CI/CDBeginnerhttps://www.udemy.com/topic/git/
MLOps SpecializationCoursera (DeepLearning.AI)4 MonthsML Fundamentals, PythonMLOps, Model Deployment, CI/CD, Model MonitoringAdvancedhttps://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops
Excel to Power Query and Power PivotUdemy5 WeeksBasic ExcelExcel, Power Query, Power Pivot, Data ModelingIntermediatehttps://www.udemy.com/topic/microsoft-power-query/
SAP FICO TrainingSLA Consultants2 MonthsNoneSAP FICO, Financial Accounting, Controlling, SAP ERPBeginner/Intermediatehttps://www.slaconsultantsindia.com/sap-fico-training
Deep Learning SpecializationCoursera (DeepLearning.AI)5 MonthsPython, ML BasicsDeep Learning, CNNs, RNNs, Neural Networks, Hyperparameter TuningAdvancedhttps://www.coursera.org/specializations/deep-learning
React - The Complete GuideUdemy10 WeeksJavaScript BasicsReact, Redux, Hooks, Context API, Frontend DevelopmentIntermediatehttps://www.udemy.com/course/react-the-complete-guide-incl-redux/
Statistics for Data ScienceCoursera (Stanford)6 WeeksBasic MathStatistics, Probability, Hypothesis Testing, RegressionBeginnerhttps://www.coursera.org/learn/stanford-statistics
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
def load_data():
    """
    Loads data ONLY from the local courses.csv file for vector embeddings.
    Raises exception if file not found or other issues.
    """
    # Load courses only from the local file
    courses_df = pd.read_csv('courses.csv')
        
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
    if courses_df is None or course_embeddings is None:
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
    RAG Query function that uses both dynamic course search (from CSV) and static text KB (from prompt).
    """
    if not llm_client:
        return "The AI Agent is not initialized. Please ensure the Gemini API key is set."
    if courses_df is None or course_embeddings is None:
        return "The course knowledge base is unavailable due to a loading error."
        
    # 1. Dynamic Course Retrieval (Vector Search)
    course_context_vector = get_rag_context(query, courses_df, course_embeddings, model)
    
    # 2. Combine all knowledge sources for the LLM
    full_context = f"""
    --- COURSE CATALOG (Vector Search Results) ---
    {course_context_vector}
    
    --- COURSE CATALOG (Static Text KB for Backup/Cross-reference) ---
    {static_kb_text}
    """
    
    rag_prompt = f"""
    You are the **PersonalAI Course Consultant** chatbot. Your goal is to answer questions about courses based *only* on the provided context (the Course Catalog). 
    
    When answering, summarize the course details, including the title, provider, duration, level, and prerequisites. Always provide the course link.
    If context does not contain the answer, state that you cannot find the information in the current catalog.

    User Query: "{query}"
    Context: {full_context}
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
    try:
        if not text.strip():
            raise ValueError("Text to convert is empty.")
            
        if engine == "edge_tts" and edge_tts is not None:
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
            st.warning("No functional TTS engine available or selected.")
            return None
            
    except Exception as e:
        st.warning(f"TTS Error: Could not generate speech. Details: {e}")
        return None

# --- STREAMLIT UI CODE ---
# Initialization of session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "tts_enabled" not in st.session_state:
    st.session_state.tts_enabled = False
if "tts_language" not in st.session_state:
    st.session_state.tts_language = DEFAULT_LANGUAGE
if "tts_engine" not in st.session_state:
    st.session_state.tts_engine = "gtts" if gTTS else ("edge_tts" if edge_tts else "None")
if "recommendations_df" not in st.session_state:
    st.session_state.recommendations_df = pd.DataFrame()


st.set_page_config(layout="wide", page_title="AI Learning Path Recommender")
st.title("💡 AI-Powered Personalized Learning Path Recommender")

# --- FIX: Initialize variables before the try block ---
COURSES_DF = None
COURSE_EMBEDDINGS = None
MODEL = None
LLM_CLIENT = None

# Load data and models
try:
    COURSES_DF, COURSE_EMBEDDINGS = load_data()
    MODEL = load_model()
    LLM_CLIENT = setup_llm()
    # Now that the loading succeeded, display the toast
    st.toast(f"Knowledge Base loaded with {len(COURSES_DF)} unique courses.", icon="🧠")
except FileNotFoundError:
    st.error("🚨 Error: 'courses.csv' not found. Please ensure the file is present in the working directory.")
except Exception as e:
    st.error(f"⚠️ Could not initialize system components. Check API keys and data files. Error: {e}.")
    
# Load Sample Profiles
try:
    with open('profiles.json', 'r') as f:
        SAMPLE_PROFILES = json.load(f)
except FileNotFoundError:
    SAMPLE_PROFILES = {}
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
    education_level = st.selectbox("Education Level:", education_options, index=education_options.index(loaded_profile.get('education_level', "Bachelor's')))
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
        if COURSES_DF is None or LLM_CLIENT is None:
            st.warning("Cannot generate path: Core system components (data or AI agent) failed to load.")
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
        if not available_engines: available_engines.append("None")

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
    
    if COURSES_DF is None or LLM_CLIENT is None:
        st.warning("Chat is disabled: Data or AI Agent failed to load during initialization.")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    if COURSES_DF is not None and LLM_CLIENT is not None:
        if prompt := st.chat_input("Ask a question about the courses in the catalog..."):
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
