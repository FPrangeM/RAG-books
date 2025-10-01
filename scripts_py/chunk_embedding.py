import duckdb
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np



# Conectar ao DuckDB
conn = duckdb.connect('./data/duckdb.db')

# conn.execute('describe lord_of_mysteries').df()

# Carregar os dados
df = conn.execute("SELECT * FROM lord_of_mysteries").df()


# --------------------------------------------------
# Divisão dos chunks otimizados para o modelo 
# --------------------------------------------------

model_name = 'intfloat/multilingual-e5-large-instruct'
max_token = 514
overlap = 50

embedded_table_name = f'lof-{model_name}'


# Configurar o divisor de texto
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=max_token-overlap,  # Tamanho aproximado de cada pedaço
    chunk_overlap=overlap,  # Overlap para manter contexto
    length_function=len,
    is_separator_regex=False,
)

# Processo de divisão
documents = []
for _, row in df.iterrows():
    capitulo = row['capitulo']
    titulo = row['titulo']
    conteudo = row['conteudo']
    
    # Dividir o conteúdo do capítulo
    chunks = text_splitter.split_text(conteudo)
    
    for i, chunk in enumerate(chunks):
        documents.append({
            'titulo': titulo,
            'capitulo': capitulo,
            'chunk_id': i,  # Identificador único do pedaço
            'conteudo': chunk,
        })

# Criar um DataFrame com os chunks
chunks_df = pd.DataFrame(documents)



# --------------------------------------------------
# Embedding 
# --------------------------------------------------


model = SentenceTransformer(model_name, device='cuda')

# Gerar embeddings para todos os chunks
texts = chunks_df['conteudo'].tolist()
embeddings = model.encode(texts, show_progress_bar=True, batch_size = 32)

# Adicionar os embeddings ao DataFrame
chunks_df['embedded'] = embeddings.tolist()


conn.execute(f'create or replace table "{embedded_table_name}" as (select * from chunks_df)')
conn.close()















conn.execute('create or replace table {embedded_table_name} as (select * from chunks_df)')
conn.close()