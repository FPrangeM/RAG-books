from sentence_transformers import SentenceTransformer
import numpy as np
import duckdb
import time


conn = duckdb.connect('./data/duckdb.db')

conn.execute('show tables').df()


embedded_table = 'lof-qwen-embedding'

embedded_df = conn.execute(f'select * from "{embedded_table}"').df()
embedded_df.head(10)



model_name = 'Qwen/Qwen3-Embedding-0.6B'


def cosine_similarity_vec(a, b):
    """Calcula cosine similarity entre dois vetores"""
    a_array = np.array(a)
    b_array = np.array(b)
    return float(np.dot(a_array, b_array) / (np.linalg.norm(a_array) * np.linalg.norm(b_array)))

# Registrar a função especificando o tipo de retorno
conn.create_function(
    'cosine_similarity',
    cosine_similarity_vec, 
    return_type='DOUBLE', 
    parameters=['DOUBLE[]', 'DOUBLE[]'])




model = SentenceTransformer(
    model_name, 
    # device='cuda',
    # model_kwargs={'attn_implementation': 'sdpa'},
    # model_kwargs={'attn_implementation': 'flash_attention_2'},
    # tokenizer_kwargs={'padding_side': 'left'},
)


pergunta = 'O que é uma singularidade ?'
query_embedding = model.encode([pergunta]).tolist()[0]


t1 = time.time()
results = conn.execute(f"""
    SELECT 
        conteudo,
        cosine_similarity(embedded, ?) as similarity
    FROM "{embedded_table}"
    ORDER BY similarity DESC
    LIMIT ?
    """
    ,[query_embedding, 3]).df()

print(f'RAG executado em {time.time()-t1:.2f}')



trechos = results.conteudo.to_list()



print(f'Considerando pura e exclusivamente os trechos a seguir:')
print()
print('--------------------------------------------------')
for i,t in enumerate(trechos):
    print(f'{i+1} -> {t}')
print('--------------------------------------------------')
print()
print(f'Responda a pergunta da forma mais completa possivel:')
print(f'{pergunta}')
