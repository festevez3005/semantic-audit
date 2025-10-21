import streamlit as st
import numpy as np
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt
from typing import List

# --- Helper Functions and Model Loading ---

@st.cache_resource 
def load_model():
    """Carga el modelo SBERT multilingüe."""
    st.write("Cargando modelo sentence-transformer multilingüe...")
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def fetch_content_from_url(url: str) -> str:
    """Extrae y limpia el contenido de texto de una URL."""
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

def chunk_document(text: str, max_chunk_length: int = 500) -> List[str]:
    """Divide el texto del documento en párrafos (chunks) basados en saltos de línea."""
    chunks = []
    current_chunk = ""
    
    # Intenta dividir por doble salto de línea (párrafos)
    paragraphs = text.split('\n\n')
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        # Si el párrafo actual excede el límite, fuerza el cierre del chunk
        if len(current_chunk) + len(paragraph) > max_chunk_length and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph
        else:
            current_chunk += " " + paragraph
            
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    # Si el chunking basado en párrafos falla y solo queda un gran chunk, usa un divisor de caracteres simple
    if len(chunks) == 1 and len(chunks[0]) > 800:
        # Intenta un método más simple de división forzada (por si el texto no tiene saltos de línea)
        chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]

    return [c.replace('\n', ' ') for c in chunks if c.strip()]


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
    # --- VALOR ACTUALIZADO AQUÍ ---
    content_input = st.text_input("Ingresar URL del Documento:", "https://www.argentina.gob.ar/justicia/convosenlaweb/situaciones/que-es-la-inteligencia-artificial")
    # -----------------------------
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

# --- VALOR ACTUALIZADO AQUÍ ---
query = st.text_input("Ingresar Consulta Objetivo:", "inteligencia artificial")
# -----------------------------

# 2. Botón de Análisis
if st.button("Calcular Relevancia Semántica", type="primary") and document_text and query:
    
    if document_text and not document_text.startswith("Error"):
        
        # ----------------------------------------------------
        # LÓGICA: CHUNKING Y BÚSQUEDA DEL MEJOR MATCH
        # ----------------------------------------------------
        chunks = chunk_document(document_text)
        if not chunks:
            st.error("No se pudo segmentar el documento en partes utilizables. Intenta con un texto más largo.")
            st.stop()
            
        st.info(f"Documento segmentado en **{len(chunks)}** partes para encontrar el mejor alineamiento semántico.")

        # 3. Generar Embeddings para todos los chunks y la query
        texts_to_embed = chunks + [query]
        with st.spinner('Generando vectores semánticos para todos los segmentos...'):
            embeddings_full = model.encode(texts_to_embed)

        query_embedding = embeddings_full[-1].reshape(1, -1)
        chunk_embeddings = embeddings_full[:-1] 

        # 4. Calcular Similitud y encontrar el score máximo
        similarity_scores = cosine_similarity(query_embedding, chunk_embeddings)[0]
        max_similarity_score = np.max(similarity_scores)
        best_chunk_index = np.argmax(similarity_scores)
        best_chunk = chunks[best_chunk_index]

        st.header("2. Resultado del Análisis Semántico")
        
        # ----------------------------------------------------
        # VISUALIZACIÓN DE LA RELEVANCIA
        # ----------------------------------------------------
        
        st.metric(label="Puntuación de Relevancia MÁXIMA (Similitud del Coseno)", value=f"{max_similarity_score:.4f}")
        
        st.subheader("Interpretación de Ingeniería de Relevancia")
        
        if max_similarity_score >= 0.7:
            st.success("✅ **Excelente Alineación.** El significado de al menos un segmento del contenido es altamente relevante para la intención de la consulta.")
        elif max_similarity_score >= 0.4:
            st.warning("⚠️ **Alineación Moderada.** Los temas están relacionados, pero el contenido podría optimizarse para mejorar el alineamiento.")
        else:
            st.error("❌ **Poca Relevancia.** El contenido está semánticamente distante de la consulta. Se necesita más contexto relevante.")

        # ----------------------------------------------------
        # 5. GRÁFICO DE DISTANCIA VECTORIAL (X e Y)
        # ----------------------------------------------------
        st.subheader("Visualización del Espacio Semántico (PCA)")
        st.markdown(f"**Comparación:** El **Mejor Segmento** ({len(best_chunk)} caracteres) vs. la **Consulta**.")
        
        # Generar embeddings SOLO para el mejor par para la visualización limpia
        embeddings_2d_pair = model.encode([best_chunk, query])

        # Aplicar PCA para reducir la dimensionalidad
        pca = PCA(n_components=2)
        vectors_2d = pca.fit_transform(embeddings_2d_pair)

        # Crear el gráfico
        fig, ax = plt.subplots()
        
        # Graficar los puntos
        ax.scatter(vectors_2d[0, 0], vectors_2d[0, 1], label='Mejor Segmento', color='blue', s=150)
        ax.scatter(vectors_2d[1, 0], vectors_2d[1, 1], label='Consulta (Intención)', color='red', s=150)
        
        # Dibujar línea
        ax.plot([vectors_2d[0, 0], vectors_2d[1, 0]], [vectors_2d[0, 1], vectors_2d[1, 1]], 
                linestyle='--', color='gray', linewidth=1)
        
        # Etiquetas
        ax.annotate("Segmento", (vectors_2d[0, 0], vectors_2d[0, 1]), textcoords="offset points", xytext=(5,5), ha='center', color='blue')
        ax.annotate("Consulta", (vectors_2d[1, 0], vectors_2d[1, 1]), textcoords="offset points", xytext=(5,5), ha='center', color='red')
        
        ax.set_title(f'Distancia Vectorial (Score Máximo: {max_similarity_score:.4f})')
        ax.legend()
        ax.grid(True)
        
        st.pyplot(fig)
        
        # Mostrar el segmento más relevante
        with st.expander("Ver Segmento del Documento Más Relevante"):
            st.markdown(f"**Párrafo (Chunk) con el Score Máximo ({max_similarity_score:.4f}):**")
            st.code(best_chunk, language='markdown')

    else:
        st.error("Asegúrate de tener contenido de documento válido y una consulta antes de ejecutar el análisis.")

# ----------------------------------------------------
# 6. PIE DE PÁGINA Y CONTACTO
# ----------------------------------------------------
st.markdown("---") 
st.markdown("""
### Información y Contacto 🤝
✨ Esta herramienta fue creada con **fines educativos y de asistencia a profesionales**.

💌 **¿Te sirvió? ¿Tenés alguna sugerencia? ¿Querés charlar sobre SEO, comunicación digital o IA aplicada?** Escribinos a: **`hola@crawla.agency`**

🌐 Conectá con Crawla en **[LinkedIn](https://www.linkedin.com/company/crawla-agency/)**
""")
