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

# --- KNOWLEDGE BASE ---
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

# --- HELPERS & DATA LOADING ---
def setup_llm():
    try:
        api_key = os.environ.get('GEMINI_API_KEY') or st.secrets.get('GEMINI_API_KEY')
        if not api_key:
            st.error("🔑 GEMINI_API_KEY not found.")
            return None
        return genai.Client(api_key=api_key)
    except Exception as e:
        st.error(f"Gemini Init Error: {e}")
        return None

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_data(kb_text):
    df = pd.read_csv(io.StringIO(kb_text))
    df = df.dropna(subset=['Title']).reset_index(drop=True)
    df.columns = df.columns.str.strip().str.lower()
    if 'skill tags' in df.columns: df = df.rename(columns={'skill tags': 'skill_tags'})
    model = load_model()
    df['search_text'] = (df['title'] + " " + df['skill_tags'] + " " + df['provider']).fillna('')
    embeddings = model.encode(df['search_text'].tolist(), show_progress_bar=False)
    return df, embeddings

def text_to_speech_conversion(text, lang_code, engine="gtts", lang_name="English"):
    """Improved TTS with safer loop handling for Streamlit"""
    # Safeguard against excessive length for voice
    if len(text) > 1500:
        text = text[:1497] + "..."
        
    try:
        if engine == "edge_tts" and edge_tts is not None:
            voice_name = EDGE_TTS_VOICE_DICT.get(lang_name, "en-US-AriaNeural")
            communicate = edge_tts.Communicate(text, voice_name)
            
            async def get_audio():
                output = b""
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio": output += chunk["content"]
                return output

            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            audio_bytes = new_loop.run_until_complete(get_audio())
            new_loop.close()
            return io.BytesIO(audio_bytes)
            
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

def run_rag_query(query, courses_df, course_embeddings, model, llm_client, kb_text):
    if not llm_client: return "AI not initialized."
    
    target_lang = st.session_state.get('tts_language', 'English')
    
    # Vector Search
    query_embed = model.encode([query])[0].reshape(1, -1)
    sims = cosine_similarity(query_embed, course_embeddings)[0]
    top_i = np.argsort(sims)[::-1][:3]
    context = ""
    for i in top_i:
        row = courses_df.iloc[i]
        context += f"Course: {row['title']} by {row['provider']}. Level: {row['level']}. Link: {row['link']}\n"

    # QUOTA-SAFE PROMPT
    rag_prompt = f"""
    You are a career consultant. Respond in {target_lang}.
    
    INSTRUCTIONS:
    1. Be concise. Stay under 120 words.
    2. Use the provided context for specific course questions.
    3. If the user asks general tech questions, be brief and encouraging.
    4. NEVER cut off mid-sentence.
    
    CONTEXT:
    {context}
    
    USER QUERY: {query}
    """
    
    try:
        # max_output_tokens=300 ensures we don't hit TPM limits or cause long lag
        response = llm_client.models.generate_content(
            model='gemini-2.0-flash', 
            contents=rag_prompt,
            config={"max_output_tokens": 400, "temperature": 0.5}
        )
        return response.text.strip()
    except Exception as e:
        return f"Error: {e}"

# --- UI LOGIC ---
st.set_page_config(layout="wide", page_title="AI Learning Path")

if "messages" not in st.session_state: st.session_state.messages = []
if "last_audio" not in st.session_state: st.session_state.last_audio = None
if "tts_language" not in st.session_state: st.session_state.tts_language = "English"

COURSES_DF, COURSE_EMBEDDINGS = load_data(KNOWLEDGE_BASE_TEXT)
MODEL = load_model()
LLM_CLIENT = setup_llm()

col_input, col_output = st.columns([1, 2])

with col_input:
    st.header("👤 Profile & Settings")
    tech_skills = st.text_area("Your Skills:", "Python, SQL")
    target_goal = st.text_input("Target Goal:", "Data Scientist")
    
    st.divider()
    st.session_state.tts_enabled = st.checkbox("Enable Voice Reply", value=True)
    st.session_state.tts_language = st.selectbox("Voice Language:", list(LANGUAGE_DICT.keys()))
    
    available_engines = []
    if gTTS: available_engines.append("gtts")
    if edge_tts: available_engines.append("edge_tts")
    st.session_state.tts_engine = st.selectbox("Engine:", available_engines)

with col_output:
    st.header("💬 Course Consultant Chat")
    
    # Display History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input
    if prompt := st.chat_input("Ask about courses..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                reply = run_rag_query(prompt, COURSES_DF, COURSE_EMBEDDINGS, MODEL, LLM_CLIENT, KNOWLEDGE_BASE_TEXT)
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
                
                if st.session_state.tts_enabled:
                    code = LANGUAGE_DICT.get(st.session_state.tts_language, "en")
                    audio = text_to_speech_conversion(reply, code, st.session_state.tts_engine, st.session_state.tts_language)
                    if audio:
                        st.session_state.last_audio = audio.getvalue()
                        st.audio(st.session_state.last_audio, format="audio/mp3", autoplay=True)

    # Audio persistence in sidebar so it doesn't disappear on click
    if st.session_state.last_audio:
        with st.sidebar:
            st.write("🔊 Replay Last Voice:")
            st.audio(st.session_state.last_audio, format="audio/mp3")
