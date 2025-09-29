from sentence_transformers import SentenceTransformer
import numpy as np
import duckdb

conn = duckdb.connect('./data/duckdb.db')

chunks_df = conn.execute('select * from lof_chunks').df()


# Carregar o modelo de embedding
model = SentenceTransformer('all-MiniLM-L6-v2')  # Modelo leve e eficaz
model = SentenceTransformer('intfloat/multilingual-e5-large', device='cuda')

# Gerar embeddings para todos os chunks
texts = chunks_df['conteudo'].tolist()
embeddings = model.encode(texts, show_progress_bar=True)

# Adicionar os embeddings ao DataFrame
chunks_df['all-MiniLM-L6-v2'] = embeddings.tolist()


conn.execute('create or replace table lof_embedded as (select * from chunks_df)')
conn.close()