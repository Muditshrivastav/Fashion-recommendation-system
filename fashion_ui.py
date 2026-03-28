import streamlit as st
import joblib
import pandas as pd
import re
import nltk
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')

# Initialize stemmer
stemmer = PorterStemmer()

def clean_and_stem(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    return ' '.join([stemmer.stem(word) for word in words])

@st.cache_resource
def load_assets():
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    matrix = joblib.load('tfidf_matrix.joblib')
    df = pd.read_csv('cleaned_men_data.csv')
    return vectorizer, matrix, df

# Load models and data
tfidf_vectorizer, tfidf_matrix, men_data = load_assets()

# Streamlit UI
st.title("Zara Men's Product Recommendation System")
st.markdown("Find similar products based on your search query using TF-IDF and Cosine Similarity.")

query = st.text_input("Enter a product category or name (e.g., 'shirt', 'trousers'):", "")
n_recommendations = st.slider("Number of recommendations:", 1, 10, 5)

if query:
    # Preprocess query
    processed_query = clean_and_stem(query)
    
    # Vectorize and calculate similarity
    query_vec = tfidf_vectorizer.transform([processed_query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Get top N indices
    top_indices = scores.argsort()[-n_recommendations:][::-1]
    results = men_data.iloc[top_indices].copy()
    results['similarity_score'] = scores[top_indices]
    
    # Display results in columns
    st.subheader(f"Top {n_recommendations} Recommendations for '{query}':")
    
    # Create a grid layout
    cols = st.columns(3)
    for i, (idx, row) in enumerate(results.iterrows()):
        col = cols[i % 3]
        with col:
            st.image(row['image_url'], use_container_width=True)
            st.write(f"**{row['product_name'].title()}**")
            st.write(f"Match: {row['similarity_score']:.2f}")
            st.divider()
else:
    st.info("Please enter a query to see recommendations.")