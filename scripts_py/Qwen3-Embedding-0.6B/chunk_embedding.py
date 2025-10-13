import duckdb
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import gc
import torch

# Conectar ao DuckDB
conn = duckdb.connect('./data/duckdb.db')

# Carregar os dados
print("Carregando dados do DuckDB...")
df = conn.execute("SELECT * FROM lord_of_mysteries").df()
print(f"Carregados {len(df)} capítulos.")

# --------------------------------------------------
# Divisão dos chunks otimizados para o modelo Qwen
# --------------------------------------------------

# ALTERAÇÃO 1: Definir o nome do novo modelo
model_name = 'Qwen/Qwen3-Embedding-0.6B'
print(f"Configurando para o modelo: {model_name}")

# ALTERAÇÃO 2: Definir um tamanho de chunk e overlap em CARACTERES,
# baseado em boas práticas para RAG (e não no limite do modelo).
# Um chunk de 512 tokens é um excelente ponto de partida.
# 512 tokens * 4 caracteres/token (aprox.) = 2048 caracteres.
chunk_size_chars = 2048
chunk_overlap_chars = 200 # ~10% do chunk_size para manter o contexto.

# O nome da tabela será atualizado automaticamente
embedded_table_name = f'lof-qwen-embedding' # Nome simplificado para a tabela


# Configurar o divisor de texto com os novos valores
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size_chars,
    chunk_overlap=chunk_overlap_chars,
    length_function=len,
    is_separator_regex=False,
)

# Processo de divisão
print("Dividindo os capítulos em chunks...")
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
            'chunk_id': f"{capitulo}_{i}",  # ID de chunk mais robusto
            'conteudo': chunk,
        })

# Criar um DataFrame com os chunks
chunks_df = pd.DataFrame(documents)
print(f"Total de {len(chunks_df)} chunks gerados.")

# --------------------------------------------------
# Embedding
# --------------------------------------------------




print("Carregando o modelo de embedding na GPU...")
# A biblioteca SentenceTransformer cuidará do download e cache

model = SentenceTransformer(
    model_name, 
    device='cuda',
    model_kwargs={'attn_implementation': 'sdpa'},
    # model_kwargs={'attn_implementation': 'flash_attention_2'},
    tokenizer_kwargs={'padding_side': 'left'},
)

gc.collect()
torch.cuda.empty_cache()

# model = SentenceTransformer(model_name, device='cuda')
print("Modelo carregado.")

# Gerar embeddings para todos os chunks
print("Gerando embeddings para todos os chunks (isso pode levar um tempo)...")
texts = chunks_df['conteudo'].tolist()

# Para documentos, o encode direto é o recomendado pela documentação do Qwen.
embeddings = model.encode(texts, show_progress_bar=True, batch_size=12)

# Adicionar os embeddings ao DataFrame
chunks_df['embedded'] = embeddings.tolist()
print("Embeddings gerados com sucesso.")

# --------------------------------------------------
# Armazenamento no Banco de Dados
# --------------------------------------------------
print(f"Salvando chunks e embeddings na tabela '{embedded_table_name}'...")
conn.execute(f'CREATE OR REPLACE TABLE "{embedded_table_name}" AS SELECT * FROM chunks_df')
conn.close()

print("\nProcesso concluído com sucesso!")