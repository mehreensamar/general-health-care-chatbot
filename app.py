import pandas as pd
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import gradio as gr 

# Load dataset
df = pd.read_csv('paumedquad.csv')  # Make sure this file is uploaded to your Space

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\W+', ' ', text)
    return text

df['Clean_Question'] = df['question'].apply(clean_text)
df['Clean_Answer'] = df['answer'].apply(clean_text)

# Load model and encode questions
model = SentenceTransformer('all-MiniLM-L6-v2')
question_embeddings = model.encode(df['Clean_Question'].tolist(), show_progress_bar=True)

# Chatbot response function
def get_response(user_query):
    user_query_clean = clean_text(user_query)
    query_embedding = model.encode([user_query_clean])
    similarities = cosine_similarity(query_embedding, question_embeddings)
    best_match_idx = np.argmax(similarities)

    matched_question = df['question'].iloc[best_match_idx]
    matched_answer = df['answer'].iloc[best_match_idx]
    confidence = similarities[0][best_match_idx]

    return f"Confidence: {confidence:.2f}\n\n{matched_answer}"

# Gradio interface
iface = gr.Interface(
    fn=get_response,
    inputs="text",
    outputs="text",
    title="Health Chatbot",
    description="Ask me anything about symptoms, diseases, or medications."
)

iface.launch(share=True)
