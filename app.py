import streamlit as st
import numpy as np
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA # 1. Importar PCA
import matplotlib.pyplot as plt # 2. Importar Matplotlib

# --- Helper Functions and Model Loading ---

@st.cache_resource 
def load_model():
    """Carga el modelo SBERT multilingüe."""
    st.write("Cargando modelo sentence-transformer multilingüe...")
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def fetch_content_from_url(url: str) -> str:
    """Extrae y limpia el contenido de texto de una URL."""
    # (Mantener la función fetch_content_from_url inalterada)
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status() 
        soup = BeautifulSoup(response.content, 'html.parser')
        text_parts = soup.find_all(['p', 'h1', 'h2', 'h3', 'li'])
        content = ' '.join([part.get_text() for part in text_parts])
        clean_content = ' '.join(content.split()) 
        return clean_content[:4096] 
    except requests.exceptions.RequestException as e:
        return f"Error al obtener la URL: {e}"
    except Exception as e:
        return f"Ocurrió un error inesperado: {e}"

# La función calculate_semantic_similarity se modificará a continuación

# --- Streamlit Application Layout ---

st.title("🌐 Herramienta Multilingüe de Ingeniería de Relevancia Semántica")
st.markdown("Mide la **alineación semántica** de tu contenido y consulta.")

# Cargar el modelo
model = load_model()

# 1. Área de Entrada de Usuario
st.header("1. Definir Contenido y Consulta (English / Español)")

source_type = st.radio(
    "Fuente del Contenido:",
    ('URL', 'Raw Text'),
    horizontal=True,
    key='source_radio'
)

document_text = ""
if source_type == 'URL':
    content_input = st.text_input("Ingresar URL del Documento:", "https://es.wikipedia.org/wiki/Inteligencia_artificial")
    if content_input:
        with st.spinner('Obteniendo y limpiando contenido...'):
            document_text = fetch_content_from_url(content_input)
            if not document_text.startswith("Error"):
                st.success(f"Contenido obtenido y limpiado ({len(document_text)} caracteres).")
            else:
                st.error(document_text)
                document_text = ""
elif source_type == 'Raw Text':
    document_text = st.text_area("Pegar Texto del Documento:", height=200, 
                                 value="La inteligencia artificial (IA) es un campo de la informática dedicado a la resolución de problemas cognitivos comúnmente asociados con la inteligencia humana.")

query = st.text_input("Ingresar Consulta Objetivo:", "Historia y evolución de los algoritmos de aprendizaje automático.")

# 2. Botón de Análisis
if st.button("Calcular Relevancia Semántica", type="primary") and document_text and query:
    
    # 3. Cálculo & Obtención de Embeddings
    if document_text and not document_text.startswith("Error"):
        
        st.header("2. Resultado del Análisis Semántico")

        # ----------------------------------------------------
        # NUEVA LÓGICA DE EMBEDDINGS Y SIMILITUD (COMBINADA)
        # ----------------------------------------------------
        
        # 3.1 Generar Embeddings
        # (Esto se hace una sola vez para eficiencia)
        with st.spinner('Generando vectores semánticos...'):
            embeddings = model.encode([document_text[:4096], query])
            doc_embedding = embeddings[0].reshape(1, -1)
            query_embedding = embeddings[1].reshape(1, -1)

        # 3.2 Calcular Similitud
        similarity_score = cosine_similarity(doc_embedding, query_embedding)[0][0]
        
        # ----------------------------------------------------
        # VISUALIZACIÓN DE LA RELEVANCIA
        # ----------------------------------------------------
        
        st.metric(label="Puntuación de Relevancia (Similitud del Coseno)", value=f"{similarity_score:.4f}")
        
        st.subheader("Interpretación de Ingeniería de Relevancia")
        
        if similarity_score >= 0.7:
            st.success("✅ **Excelente Alineación.** El significado del contenido es altamente relevante para la intención de la consulta.")
        elif similarity_score >= 0.4:
            st.warning("⚠️ **Alineación Moderada.** Los temas están relacionados, pero el contenido podría optimizarse.")
        else:
            st.error("❌ **Poca Relevancia.** El contenido está semánticamente distante de la consulta.")

        # ----------------------------------------------------
        # 4. GRÁFICO DE DISTANCIA VECTORIAL (X e Y)
        # ----------------------------------------------------
        st.subheader("Visualización del Espacio Semántico (PCA)")
        st.markdown("La distancia se proyecta de **384 dimensiones a 2 dimensiones** (ejes X e Y) para visualizar la cercanía de los conceptos.")
        
        # Aplicar PCA para reducir la dimensionalidad de 384D a 2D
        pca = PCA(n_components=2)
        vectors_2d = pca.fit_transform(embeddings)

        # Crear el gráfico
        fig, ax = plt.subplots()
        
        # Graficar los puntos
        ax.scatter(vectors_2d[0, 0], vectors_2d[0, 1], label='Documento (Contenido)', color='blue', s=100)
        ax.scatter(vectors_2d[1, 0], vectors_2d[1, 1], label='Consulta (Intención)', color='red', s=100)
        
        # Dibujar una línea entre ellos para ilustrar la distancia
        ax.plot([vectors_2d[0, 0], vectors_2d[1, 0]], [vectors_2d[0, 1], vectors_2d[1, 1]], 
                linestyle='--', color='gray', linewidth=1)
        
        # Añadir etiquetas
        ax.annotate("Documento", (vectors_2d[0, 0], vectors_2d[0, 1]), textcoords="offset points", xytext=(5,5), ha='center', color='blue')
        ax.annotate("Consulta", (vectors_2d[1, 0], vectors_2d[1, 1]), textcoords="offset points", xytext=(5,5), ha='center', color='red')
        
        ax.set_title(f'Distancia Vectorial Semántica (Score: {similarity_score:.4f})')
        ax.legend()
        ax.grid(True)
        
        # Mostrar el gráfico en Streamlit
        st.pyplot(fig)
        
        # Muestra el código de vector slice como antes
        with st.expander("Mostrar Segmentos de Vector Original"):
            st.markdown("*(Los vectores de alta dimensión que fueron proyectados)*")
            doc_vector_slice = embeddings[0][:5]
            query_vector_slice = embeddings[1][:5]
            st.code(f"Segmento del Vector del Documento: {doc_vector_slice}", language='python')
            st.code(f"Segmento del Vector de la Consulta: {query_vector_slice}", language='python')
    else:
        st.error("Asegúrate de tener contenido de documento válido y una consulta antes de ejecutar el análisis.")
