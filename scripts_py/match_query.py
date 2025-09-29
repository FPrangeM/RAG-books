from sentence_transformers import SentenceTransformer
import numpy as np
import duckdb

conn = duckdb.connect('./data/duckdb.db')

embedded_df = conn.execute('select * from lof_embedded').df()
embedded_df.head(2)
# output:
'''
	titulo	capitulo	chunk_id	conteudo	all-MiniLM-L6-v2
0	Carmesim	1	0	“Dói!”\n[1]\n“Dói tanto!”\n“Minha cabeça dói d...	[-0.05513400584459305, 0.03674433380365372, 0....
1	Carmesim	1	1	Zhou Mingrui, que não desconhecia encontros si...	[-0.014141588471829891, 0.016819780692458153, ...
'''




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








model = SentenceTransformer('all-MiniLM-L6-v2')


pergunta = 'Qual a formula da poção do palhaço do caminho do vidente?'
query_embedding = model.encode([pergunta]).tolist()[0]


results = conn.execute("""
    SELECT capitulo, titulo, conteudo, chunk_id,
            cosine_similarity("all-MiniLM-L6-v2", ?) as similarity
    FROM lof_embedded
    ORDER BY similarity DESC
    LIMIT ?
""", [query_embedding, 3]).df()



trechos = results.conteudo.to_list()



print(f'Considerando pura e exclusivamente os trechos a seguir: ')
print(f'{trechos}\n')
print(f'Responda a pergunta:')
print(f'{pergunta}')
