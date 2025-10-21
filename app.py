import streamlit as st
import numpy as np
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List

# --- Helper Functions and Model Loading ---

@st.cache_resource 
def load_model():
    """
    Loads the multilingual SBERT model into cache.
    'paraphrase-multilingual-MiniLM-L12-v2' supports over 50 languages, 
    including English and Spanish, by mapping them into a shared semantic space.
    """
    st.write("Loading multilingual sentence-transformer model...")
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def fetch_content_from_url(url: str) -> str:
    """Fetches and cleans text content from a given URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status() 
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract text from common elements
        text_parts = soup.find_all(['p', 'h1', 'h2', 'h3', 'li'])
        content = ' '.join([part.get_text() for part in text_parts])
        
        # Simple cleanup and normalization
        clean_content = ' '.join(content.split()) 
        
        # Limit content length for efficiency, models can handle up to ~512 tokens
        return clean_content[:4096] 

    except requests.exceptions.RequestException as e:
        return f"Error fetching URL: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

def calculate_semantic_similarity(doc_text: str, query: str, model: SentenceTransformer) -> float:
    """
    Generates embeddings for the document and query, then calculates Cosine Similarity.
    """
    if not doc_text or not query:
        return 0.0

    # Encode both the document and the query into vectors
    # This vector represents the AI's understanding (semantic space)
    embeddings: np.ndarray = model.encode([doc_text, query])
    
    doc_embedding: np.ndarray = embeddings[0].reshape(1, -1)
    query_embedding: np.ndarray = embeddings[1].reshape(1, -1)
    
    # Calculate Cosine Similarity (Relevance Score)
    similarity_score: np.ndarray = cosine_similarity(doc_embedding, query_embedding)
    
    return similarity_score[0][0]

# --- Streamlit Application Layout ---

st.title("🌐 Multilingual Semantic Relevance Tool")
st.markdown("Measures the **semantic alignment** between your content (English or Spanish) and a target query, using a multilingual model.")

# Load the model
model = load_model()

# 1. User Input Area
st.header("1. Define Content & Query (English / Español)")

source_type = st.radio(
    "Content Source:",
    ('URL', 'Raw Text'),
    horizontal=True,
    key='source_radio'
)

document_text = ""
content_input = ""

if source_type == 'URL':
    content_input = st.text_input("Enter Document URL:", "https://es.wikipedia.org/wiki/Inteligencia_artificial")
    if content_input:
        with st.spinner('Fetching and cleaning content...'):
            document_text = fetch_content_from_url(content_input)
            if not document_text.startswith("Error"):
                st.success(f"Content fetched and cleaned ({len(document_text)} characters).")
            else:
                st.error(document_text)
                document_text = "" # Clear to prevent running analysis
                
elif source_type == 'Raw Text':
    content_input = st.text_area("Paste Document Text:", height=200, 
                                 value="La inteligencia artificial (IA) es un campo de la informática dedicado a la resolución de problemas cognitivos comúnmente asociados con la inteligencia humana.")
    document_text = content_input

query = st.text_input("Enter Target Query:", "Historia y evolución de los algoritmos de aprendizaje automático.")

# 2. Analysis Button
if st.button("Calculate Semantic Similarity", type="primary") and document_text and query:
    
    # Run analysis only if content and query are valid
    if document_text and not document_text.startswith("Error"):
        
        # 3. Calculation & Output
        st.header("2. Semantic Analysis Result")
        
        # Calculate the similarity
        score = calculate_semantic_similarity(document_text, query, model)
        
        st.metric(label="Relevance Score (Cosine Similarity)", value=f"{score:.4f}")
        
        st.subheader("Relevance Engineering Interpretation")
        
        # Interpret the score
        if score >= 0.7:
            st.success("✅ **Excelente Alineación / Excellent Alignment.** The content's meaning is highly relevant to the query's intent.")
        elif score >= 0.4:
            st.warning("⚠️ **Alineación Moderada / Moderate Alignment.** The topics are related, but the content should be optimized to better match the specific semantic space of the query.")
        else:
            st.error("❌ **Poca Relevancia / Poor Relevance.** The content is semantically distant from the query. Your content is likely off-topic for this search intent.")

        st.subheader("AI's Interpretation (Vector Check)")
        
        # Optional: Show vector representation
        with st.expander("Show Multilingual Vector Slices"):
            embeddings = model.encode([document_text, query])
            doc_vector_slice = embeddings[0][:5]
            query_vector_slice = embeddings[1][:5]
            
            st.markdown("*(Note: The AI's 'understanding' is represented by the position of these vectors in the semantic space. Close vectors = High Relevance)*")
            st.code(f"Document Vector Slice: {doc_vector_slice}", language='python')
            st.code(f"Query Vector Slice:    {query_vector_slice}", language='python')
    else:
        st.error("Please ensure you have valid document content and a query before running the analysis.")
