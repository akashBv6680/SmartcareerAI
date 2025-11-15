import streamlit as st
import pandas as pd
import numpy as np
import json
import os # For environment variables
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai # For Gemini API

# --- 0. LLM CONFIGURATION ---

def setup_llm():
    """Initializes the Gemini client."""
    # Assumes GEMINI_API_KEY is set in environment variables (or Streamlit secrets)
    try:
        if 'GEMINI_API_KEY' not in os.environ:
             st.error("🔑 Error: GEMINI_API_KEY environment variable not found. Please set it up in Streamlit secrets.")
             return None
        return genai.Client()
    except Exception as e:
        st.error(f"Failed to initialize Gemini Client: {e}")
        return None

# --- 1. CONFIGURATION AND INITIALIZATION ---

@st.cache_resource
def load_model():
    """Load the Sentence Transformer model (cached for performance)."""
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_data():
    """Load course catalog and generate embeddings (cached)."""
    try:
        courses_df = pd.read_csv('courses.csv')
    except FileNotFoundError:
        st.error("Error: 'courses.csv' not found. Please create the file with the required columns.")
        st.stop()
        
    model = load_model()
    
    # Concatenate relevant course text for embedding
    courses_df['search_text'] = (
        courses_df['title'] + " " + courses_df['skill_tags'] + " " + 
        courses_df['provider'] + " " + courses_df['level']
    )
    
    # Generate embeddings for courses
    course_embeddings = model.encode(courses_df['search_text'].tolist(), show_progress_bar=False)
    
    return courses_df, course_embeddings

def generate_user_embedding(user_profile, model):
    """Generate a single embedding for the user profile."""
    profile_text = (
        f"Education: {user_profile['education_level']} in {user_profile['major']}. "
        f"Technical Skills: {user_profile['technical_skills']}. "
        f"Soft Skills: {user_profile['soft_skills']}. "
        f"Goal Domain: {user_profile['target_domain']}."
    )
    return model.encode([profile_text])[0].reshape(1, -1)

def map_prerequisite_level(level_str):
    """Maps level string to a numerical value for comparison."""
    if pd.isna(level_str): return 0
    mapping = {'None': 0, 'Beginner': 1, 'Intermediate': 2, 'Advanced': 3}
    return mapping.get(level_str.strip(), 0)

def map_course_level(level_str):
    """Maps course level to a numerical value."""
    mapping = {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3}
    return mapping.get(level_str.strip(), 0)

def generate_llm_rationale(client, user_profile, course_row, timeline_type):
    """Generates the justification text using the Gemini LLM."""
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
    """Main function to compute similarity, rank, and filter courses."""
    
    # 1. Get user embedding
    user_embed = generate_user_embedding(user_profile, model)
    
    # 2. Calculate Cosine Similarity
    similarity_scores = cosine_similarity(user_embed, course_embeddings)[0]
    
    results_df = courses_df.copy()
    results_df['similarity_score'] = similarity_scores
    
    # 3. Prerequisite and Level Matching Logic (Filtering/Penalty)
    
    # Heuristic for user's starting level
    user_level = 1 # Beginner
    if 'intermediate' in user_profile['technical_skills'].lower() or user_profile['education_level'] in ['Master\'s', 'PhD']:
        user_level = 2
    if 'advanced' in user_profile['technical_skills'].lower() or user_profile['education_level'] == 'PhD':
        user_level = 3
        
    results_df['course_level_num'] = results_df['level'].apply(map_course_level)
    
    # CORRECTION APPLIED HERE (Handling NaN in prerequisites)
    results_df['prereq_level_num'] = results_df['prerequisites'].apply(
        lambda x: 0 if pd.isna(x) else map_prerequisite_level(str(x).split(',')[0].strip())
    )

    # Penalty for recommending too-advanced courses:
    results_df['prereq_penalty'] = np.where(
        results_df['prereq_level_num'] > user_level, 0.5, 1.0 
    )
    
    # Final Fit Score (0-100)
    results_df['fit_score'] = (results_df['similarity_score'] * 100 * results_df['prereq_penalty']).round(1)
    
    # 4. Rank and Filter
    ranked_courses = results_df.sort_values(by='fit_score', ascending=False).head(10).copy()
    
    # 5. Generate Timeline
    def assign_timeline(row):
        is_basic = row['level'] in ['Beginner', 'Intermediate']
        duration_lower = row['duration'].lower()
        is_short = 'week' in duration_lower or ('month' in duration_lower and int(duration_lower.split()[0]) <= 2)
        
        if is_basic and is_short and row['fit_score'] >= 50:
            return 'Short-Term'
        elif row['fit_score'] >= 40:
            return 'Long-Term'
        return 'Long-Term'
        
    ranked_courses['timeline'] = ranked_courses.apply(assign_timeline, axis=1)

    # 6. Generate LLM Rationale
    ranked_courses['rationale'] = ranked_courses.apply(
        lambda row: generate_llm_rationale(llm_client, user_profile, row, row['timeline']), axis=1
    )

    return ranked_courses

def format_json_output(df, profile_name):
    """Formats the recommendation DataFrame into the required JSON structure."""
    
    short_term_courses = df[df['timeline'] == 'Short-Term'].to_dict('records')
    long_term_courses = df[df['timeline'] == 'Long-Term'].to_dict('records')

    def serialize_record(record):
        return {
            'title': record['title'],
            'provider': record['provider'],
            'duration': record['duration'],
            'level': record['level'],
            'fit_score': float(record['fit_score']), 
            'rationale': record['rationale'],
            'link': record['link']
        }

    return {
        "profile_name": profile_name,
        "learning_path": {
            "short_term": {
                "timeline": "Next 1-3 Months: Foundational/High-Priority Courses",
                "rationale": "Focus on high-impact, foundational skills that immediately address current gaps.",
                "courses": [serialize_record(c) for c in short_term_courses]
            },
            "long_term": {
                "timeline": "Next 3-12 Months: Specialization/Advanced Courses",
                "rationale": "Focus on deeper specialization and advanced topics to achieve the final career goal.",
                "courses": [serialize_record(c) for c in long_term_courses]
            }
        }
    }


# --- 2. STREAMLIT UI CODE ---

st.set_page_config(layout="wide", page_title="AI Learning Path Recommender")

st.title("💡 AI-Powered Personalized Learning Path Recommender")
st.markdown("""
A smart recommender that suggests suitable college programs, certifications, or curated learning paths based on your profile, skills, and goals.
""")

# Load data and model
try:
    COURSES_DF, COURSE_EMBEDDINGS = load_data()
    MODEL = load_model()
    LLM_CLIENT = setup_llm() # Initialize Gemini Client
except Exception as e:
    st.error(f"Could not load data or model: {e}. Check dependencies.")
    st.stop()

# Load sample profiles
try:
    with open('profiles.json', 'r') as f:
        SAMPLE_PROFILES = json.load(f)
except FileNotFoundError:
    SAMPLE_PROFILES = {}
    st.warning("Could not find 'profiles.json'. Using manual input only.")


# --- SIDEBAR (USER INPUT) ---

with st.sidebar:
    st.header("👤 User Profile Input")
    
    profile_keys = ["Manual Input"] + list(SAMPLE_PROFILES.keys())
    profile_selection = st.selectbox(
        "Load Sample Profile:",
        profile_keys
    )
    
    if profile_selection != "Manual Input":
        loaded_profile = SAMPLE_PROFILES[profile_selection]
    else:
        loaded_profile = {}

    
    # 1. Required Inputs
    st.subheader("Required Background")
    
    education_options = ["Bachelor's", "Master's", "PhD", "High School/GED", "Certificate"]
    education_level = st.selectbox(
        "Education Level:",
        education_options,
        index=education_options.index(loaded_profile.get('education_level', "Bachelor's"))
    )
    
    major = st.text_input(
        "Major/Degree:",
        value=loaded_profile.get('major', "Computer Science")
    )
    
    technical_skills = st.text_area(
        "Technical Skills (comma separated):",
        value=loaded_profile.get('technical_skills', "Python, SQL, Data Analysis, Excel, Git")
    )
    
    soft_skills = st.text_area(
        "Soft Skills (comma separated):",
        value=loaded_profile.get('soft_skills', "Communication, Leadership, Problem-Solving")
    )
    
    # 2. Optional Inputs
    st.subheader("Goals & Preferences (Optional)")
    
    target_domain = st.text_input(
        "Target Career Domain (e.g., Data Science, UX Design, DevOps):",
        value=loaded_profile.get('target_domain', "Data Science")
    )
    
    duration_options = ["Short-term (1-3 months)", "Long-term (3-12 months)", "Any"]
    loaded_duration = loaded_profile.get('preferred_duration', "Any")
    
    # FIX FOR VALUE ERROR: Find the index of the option containing the loaded value
    duration_index = next(
        (i for i, opt in enumerate(duration_options) if loaded_duration in opt), 
        duration_options.index("Any") # Default to "Any" if not found
    )
    
    preferred_duration = st.selectbox(
        "Preferred Study Duration:",
        duration_options,
        index=duration_index # Use the safely determined index
    )
    
    # Final User Profile Dict
    USER_PROFILE = {
        'education_level': education_level,
        'major': major,
        'technical_skills': technical_skills,
        'soft_skills': soft_skills,
        'target_domain': target_domain,
        'preferred_duration': preferred_duration, # Now uses the full string option
    }

    st.markdown("---")
    st.info("Click 'Generate Recommendations' to begin.")


# --- MAIN CONTENT AREA ---

if st.button("🚀 Generate Recommendations", type="primary"):
    
    if not USER_PROFILE['technical_skills'].strip() or not USER_PROFILE['target_domain'].strip():
        st.error("Please provide at least your Technical Skills and Target Career Domain.")
        st.stop()
        
    with st.spinner("Analyzing profile, computing similarity, and generating LLM rationale..."):
        # Run the matching engine
        recommendations_df = recommend_courses(
            USER_PROFILE, 
            COURSES_DF, 
            COURSE_EMBEDDINGS, 
            MODEL,
            LLM_CLIENT # Pass the Gemini client
        )
    
    st.header(f"🎯 Recommended Learning Path for **{target_domain}**")
    st.markdown(f"**Current Technical Skills:** {USER_PROFILE['technical_skills']}")
    
    
    # --- SHORT-TERM PLAN ---
    
    st.subheader("🗓️ Short-Term Plan (Next 1-3 Months)")
    st.caption("Focus on foundational, high-impact skills with fast turnaround.")
    
    short_term = recommendations_df[recommendations_df['timeline'] == 'Short-Term']
    
    if not short_term.empty:
        for i, row in short_term.iterrows():
            st.success(f"**{row['title']}** ({row['provider']})")
            
            col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
            col1.metric("Fit Score", f"{row['fit_score']}%")
            col2.metric("Level", row['level'])
            col3.metric("Duration", row['duration'])
            col4.markdown(f"**Rationale:** {row['rationale']}")
            st.markdown(f"**Enroll:** [Access Course Link Here]({row['link']})")
            st.markdown("---")
    else:
        st.info("No courses prioritized for the short term based on current criteria. Moving to Long-Term focus.")


    # --- LONG-TERM PLAN ---
    
    st.subheader("📚 Long-Term Plan (Next 3-12 Months)")
    st.caption("Focus on deep specialization, advanced certifications, and career goal achievement.")
    
    long_term = recommendations_df[recommendations_df['timeline'] == 'Long-Term']

    if not long_term.empty:
        for i, row in long_term.iterrows():
            st.info(f"**{row['title']}** ({row['provider']})")
            
            col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
            col1.metric("Fit Score", f"{row['fit_score']}%")
            col2.metric("Level", row['level'])
            col3.metric("Duration", row['duration'])
            col4.markdown(f"**Rationale:** {row['rationale']}")
            st.markdown(f"**Enroll:** [Access Course Link Here]({row['link']})")
            st.markdown("---")
    else:
        st.info("No courses recommended for the long term.")
    
    
    # --- JSON OUTPUT ---
    st.subheader("💾 JSON Output (for API Integration)")
    
    json_output = format_json_output(
        recommendations_df, 
        profile_selection if profile_selection != "Manual Input" else "Custom User Profile"
    )
    
    st.json(json_output)
    
# --- END OF APP ---
