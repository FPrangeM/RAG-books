from sentence_transformers import SentenceTransformer
import numpy as np
import duckdb



conn = duckdb.connect('./data/duckdb.db')

conn.execute('show tables').df()


embedded_table = 'lof-intfloat/multilingual-e5-large-instruct'

embedded_df = conn.execute(f'select * from "{embedded_table}"').df()
embedded_df.head(2)



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




model = SentenceTransformer('intfloat/multilingual-e5-large-instruct', device='cuda')

# model = SentenceTransformer('intfloat/multilingual-e5-large')
# model = SentenceTransformer('all-MiniLM-L6-v2')


pergunta = 'O que é o MI9 ?'
query_embedding = model.encode([pergunta]).tolist()[0]


results = conn.execute(f"""
    SELECT 
        conteudo,
        cosine_similarity(embedded, ?) as similarity
    FROM "{embedded_table}"
    ORDER BY similarity DESC
    LIMIT ?
    """
    ,[query_embedding, 20]).df()



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
