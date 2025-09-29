import duckdb
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd

# Conectar ao DuckDB
conn = duckdb.connect('./data/duckdb.db')

# conn.execute('describe lord_of_mysteries').df()

# Carregar os dados
df = conn.execute("SELECT * FROM lord_of_mysteries").df()

# Configurar o divisor de texto
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Tamanho aproximado de cada pedaço
    chunk_overlap=200,  # Overlap para manter contexto
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

conn.execute('create or replace table lof_chunks as (select * from chunks_df)')
conn.close()