import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
import os  # Import the os module
from typing import Dict,List

# --- Helper Functions --

def get_embedding(text, model="text-embedding-3-small", api_key=None):
    """
    Gets the embedding for a given text using the OpenAI API.
    """
    if api_key is None:
        api_key = "xxxx"

    url = "xxxxx"
    headers = {
        "Authorization":  f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "input": text,
        "model": model,
        "stream": True}
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()["data"][0]["embedding"]
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        return None
    except (KeyError, IndexError) as e:
        st.error(f"Error parsing API response: {e}. Response: {response.text}")
        return None


def calculate_similarity_matrix(sentences, topics, model_name, api_key):
    """
    Calculates the similarity matrix between sentences and topics.
    """
    sentence_embeddings = []
    topic_embeddings = []

    # Get embeddings for sentences
    for sentence in sentences:
        embedding = get_embedding(sentence, model=model_name, api_key=api_key)
        if embedding:
            sentence_embeddings.append(embedding)
        else:
            return None  # Return None if any embedding fails

    # Get embeddings for topics
    for topic in topics:
        embedding = get_embedding(topic, model=model_name, api_key=api_key)
        if embedding:
            topic_embeddings.append(embedding)
        else:
            return None  # Return None if any embedding fails

    if not sentence_embeddings or not topic_embeddings:
        st.warning("Could not generate embeddings for all inputs. Please check your inputs and API key.")
        return None

    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(sentence_embeddings, topic_embeddings)

    # Convert to percentages
    similarity_matrix_percentage = similarity_matrix * 100

    return similarity_matrix_percentage


def color_scale(value):
    """
    Generates a color based on the similarity score.
    """
    if value >= 80:
        return "background-color: #008000"  # Green
    elif value >= 60:
        return "background-color: #66B066"  # Light Green
    elif value >= 40:
        return "background-color: #FFC300"  # Yellow
    elif value >= 20:
        return "background-color: #FF5733"  # Orange
    else:
        return "background-color: #C70039"  # Red


# --- Streamlit App ---

st.title("Sentence-Topic Similarity Analyzer")

# --- Input Fields ---
with st.form("input_form"):
    st.subheader("Enter Text and Topics")
    text_input = st.text_area("Enter text (one sentence per line):", height=150)
    topic_input = st.text_area("Enter topics (one topic per line):", height=100)

    # Model Selection
    model_name = st.selectbox("Select Model:", ["text-embedding-3-small"])

    # API Key Input (Optional, but recommended for security)

    submitted = st.form_submit_button("Analyze Similarity")

if submitted:
    # --- Data Processing ---
    sentences = [s.strip() for s in text_input.splitlines() if s.strip()]  # Split into sentences, remove empty lines
    topics = [t.strip() for t in topic_input.splitlines() if t.strip()]  # Split into topics, remove empty lines

    if not sentences or not topics:
        st.warning("Please enter both text and topics.")
    else:
        similarity_matrix = calculate_similarity_matrix(sentences, topics, model_name, "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImFrdWxhLmtyaXNobmFAc3RyYWl2ZS5jb20ifQ.R14R_Hw_Hp8ut1Y53pVvAKhULwP4-oIp469jsns4hMY")

        if similarity_matrix is not None:
            # --- Visualization ---
            st.subheader("Similarity Matrix")
            import pandas as pd

            df = pd.DataFrame(similarity_matrix, index=sentences, columns=topics)
            styled_df = df.style.applymap(color_scale)
            st.dataframe(styled_df)
