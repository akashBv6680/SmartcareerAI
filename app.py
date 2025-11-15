import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

# --- 1. CONFIGURATION AND INITIALIZATION ---

@st.cache_resource
def load_model():
    """Load the Sentence Transformer model (cached for performance)."""
    # Using a fast, lightweight model for demonstration
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_data():
    """Load course catalog and generate embeddings (cached)."""
    try:
        courses_df = pd.read_csv('courses.csv')
    except FileNotFoundError:
        st.error("Error: 'courses.csv' not found. Please create the file.")
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

def determine_recommended_prep(user_skills, course_prerequisites):
    """Simple check to suggest preparation."""
    # Placeholder for more complex NLP-based prep check
    if course_prerequisites.lower() != 'none':
        return f"Prep: Ensure you have strong foundation in **{course_prerequisites}**."
    return ""

def generate_rationale_and_gap(user_profile, course_row):
    """Generates the justification text using a simple prompt-like structure."""
    
    user_skills = user_profile['technical_skills'].lower().split(', ')
    course_skills = course_row['skill_tags'].lower().split(', ')
    target_domain = user_profile['target_domain']
    
    # 1. Matching
    matched_skills = [s.capitalize() for s in user_skills if s in course_skills]
    match_str = f"This course aligns with your current skills in **{', '.join(matched_skills[:2])}**." if matched_skills else ""
    
    # 2. Gap Filling (what new skill does it offer)
    new_skills = [s.capitalize() for s in course_skills if s not in user_skills]
    gap_str = f"It will fill the gap by teaching you **{', '.join(new_skills[:2])}** needed for a role in **{target_domain}**."
    
    # 3. Preparation
    prep_str = determine_recommended_prep(user_profile['technical_skills'], course_row['prerequisites'])
    
    return f"{match_str} {gap_str} {prep_str}".strip()

def recommend_courses(user_profile, courses_df, course_embeddings, model):
    """Main function to compute similarity, rank, and filter courses."""
    
    # 1. Get user embedding
    user_embed = generate_user_embedding(user_profile, model)
    
    # 2. Calculate Cosine Similarity
    # Similarity score is between 0 and 1
    similarity_scores = cosine_similarity(user_embed, course_embeddings)[0]
    
    results_df = courses_df.copy()
    results_df['similarity_score'] = similarity_scores
    
    # 3. Prerequisite and Level Matching Logic (Filtering/Penalty)
    
    # Convert user and course levels to numerical values
    # For simplification, we'll assume a user is at a level that is roughly 
    # Intermediate (2) if they have a degree, and Advanced (3) if they have a 
    # degree + many skills. Here, we use a simple heuristic:
    user_level = 1 # Start at Beginner
    if 'intermediate' in user_profile['technical_skills'].lower() or user_profile['education_level'] in ['Master\'s', 'PhD']:
        user_level = 2
    if 'advanced' in user_profile['technical_skills'].lower() or user_profile['education_level'] == 'PhD':
        user_level = 3
        
    results_df['course_level_num'] = results_df['level'].apply(map_course_level)
    results_df['prereq_level_num'] = results_df['prerequisites'].apply(lambda x: map_prerequisite_level(x.split(',')[0].strip())) 
    
    # Penalty for recommending too-advanced courses:
    # If course level is 2 levels higher than user's heuristic level, significantly penalize score
    # e.g., Beginner (1) recommending Advanced (3)
    results_df['prereq_penalty'] = np.where(
        results_df['prereq_level_num'] > user_level, 0.5, 1.0 
    )
    
    # Final Fit Score (0-100)
    results_df['fit_score'] = (results_df['similarity_score'] * 100 * results_df['prereq_penalty']).round(1)
    
    # 4. Rank and Filter
    ranked_courses = results_df.sort_values(by='fit_score', ascending=False).head(10)
    
    # 5. Generate Rationale and Timeline
    ranked_courses['rationale'] = ranked_courses.apply(
        lambda row: generate_rationale_and_gap(user_profile, row), axis=1
    )
    
    # Simple Timeline Logic: 
    # High fit_score, Beginner/Intermediate, and short duration go to Short-Term.
    # Lower fit_score, Advanced, and long duration go to Long-Term.
    def assign_timeline(row):
        is_basic = row['level'] in ['Beginner', 'Intermediate']
        is_short = 'week' in row['duration'].lower() or ('month' in row['duration'].lower() and int(row['duration'].split()[0]) <= 2)
        
        # Priority: Basic/Short courses with good fit score
        if is_basic and is_short and row['fit_score'] >= 50:
            return 'Short-Term'
        # Secondary: More advanced or longer courses
        elif row['fit_score'] >= 40:
            return 'Long-Term'
        return 'Long-Term' # Default
        
    ranked_courses['timeline'] = ranked_courses.apply(assign_timeline, axis=1)

    return ranked_courses

def format_json_output(df, profile_name):
    """Formats the recommendation DataFrame into the required JSON structure."""
    
    short_term_courses = df[df['timeline'] == 'Short-Term'].to_dict('records')
    long_term_courses = df[df['timeline'] == 'Long-Term'].to_dict('records')

    # Convert object to string/float for JSON serialization
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
except Exception as e:
    st.error(f"Could not load data or model: {e}")
    st.stop()

# Load sample profiles
try:
    with open('profiles.json', 'r') as f:
        SAMPLE_PROFILES = json.load(f)
except FileNotFoundError:
    SAMPLE_PROFILES = {}
    st.warning("Could not find 'profiles.json'. Please manually input data.")


# --- SIDEBAR (USER INPUT) ---

with st.sidebar:
    st.header("👤 User Profile Input")
    
    profile_selection = st.selectbox(
        "Load Sample Profile:",
        ["Manual Input"] + list(SAMPLE_PROFILES.keys())
    )
    
    if profile_selection != "Manual Input":
        loaded_profile = SAMPLE_PROFILES[profile_selection]
    else:
        loaded_profile = {}

    
    # 1. Required Inputs
    st.subheader("Required Background")
    
    education_level = st.selectbox(
        "Education Level:",
        ["Bachelor's", "Master's", "PhD", "High School/GED", "Certificate"],
        index=["Bachelor's", "Master's", "PhD", "High School/GED", "Certificate"].index(loaded_profile.get('education_level', "Bachelor's"))
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
    
    preferred_duration = st.selectbox(
        "Preferred Study Duration:",
        ["Short-term (1-3 months)", "Long-term (3-12 months)", "Any"],
        index=["Short-term (1-3 months)", "Long-term (3-12 months)", "Any"].index(loaded_profile.get('preferred_duration', "Any"))
    )
    
    # Final User Profile Dict
    USER_PROFILE = {
        'education_level': education_level,
        'major': major,
        'technical_skills': technical_skills,
        'soft_skills': soft_skills,
        'target_domain': target_domain,
        'preferred_duration': preferred_duration,
    }

    st.markdown("---")
    st.info("Click 'Generate Recommendations' to begin.")


# --- MAIN CONTENT AREA ---

if st.button("🚀 Generate Recommendations", type="primary"):
    
    with st.spinner("Analyzing profile and computing similarity..."):
        # Run the matching engine
        recommendations_df = recommend_courses(
            USER_PROFILE, 
            COURSES_DF, 
            COURSE_EMBEDDINGS, 
            MODEL
        )
    
    st.header(f"🎯 Recommended Learning Path for **{target_domain}**")
    st.write(f"**Current Skills:** {USER_PROFILE['technical_skills']}")
    
    
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
        st.info("No courses prioritized for the short term based on current criteria.")


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
