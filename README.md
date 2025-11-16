# 🚀 AI-Powered Personalized Learning Path Recommender

## Project Overview

The **AI Learning Path Recommender** is a Streamlit application designed to generate customized, multi-stage learning paths (Short-Term and Long-Term) for users based on their existing skills and target career domain.

It leverages two core functionalities:
1.  **Vector-based Course Recommendation:** Uses Sentence Transformers and Cosine Similarity to match user profiles (skills, major, goal) to a detailed course catalog (`courses.csv`).
2.  **Hybrid RAG Chatbot:** An intelligent agent built with the Gemini API that can answer specific questions about the course catalog using a **Knowledge Base Retrieval Tool** and answer general technology questions using its **General LLM Knowledge**.

This project provides a robust framework for delivering personalized educational content and intelligent Q\&A.

---

## ✨ Key Features & Performance

| Feature | Description | Performance Highlight |
| :--- | :--- | :--- |
| **Dual-Tool RAG Agent** | Explicitly instructs the AI to use either the **Course Catalog (Retrieval Tool)** for catalog specifics or its **General Knowledge (LLM)** for concepts (e.g., "What is Docker?"). | Improves response accuracy and scope by disambiguating the source of information. |
| **Personalized Path Generation** | Ranks courses by **Fit Score** based on skill match and prerequisite adherence, generating distinct Short-Term and Long-Term plans. | High relevance, ensuring users tackle foundational knowledge before advanced topics. |
| **LLM-Powered Rationale** | Uses the Gemini API to provide a **concise, two-sentence rationale** for *every* course recommendation, explaining the skill gap it addresses. | Enhances user trust and understanding of the recommendation logic. |
| **Multi-Engine Text-to-Speech (TTS)** | Supports both **gTTS** and **Edge-TTS** (if libraries are installed) for an accessible, audio-based response experience in multiple languages. | Increases application accessibility and multilingual support. |
| **Scalable Data Structure** | Uses an embedded course catalog in Python (`KNOWLEDGE_BASE_TEXT`) and an external `courses.csv` for flexible, vectorized search indexing. | Maintains fast retrieval speed while allowing easy data updates via the CSV file. |

---

## 🛠️ Setup and Installation

### Prerequisites

1.  **Python 3.8+**
2.  **A Gemini API Key** (obtainable from Google AI Studio).

### 1. Environment Setup

```bash
# Clone the repository (if applicable)
# git clone <your-repo-url>
# cd smartcareerai

# Create a virtual environment and activate it
python -m venv venv
source venv/bin/activate  # On Linux/macOS
# .\venv\Scripts\activate # On Windows
