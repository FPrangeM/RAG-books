# given a interval of chapters, this script will scrap and save the chapter, title and content to a database.
# the database save will occur in batches of 100 register, while the scrap will occur in interval of 20.

import asyncio
import aiohttp
from bs4 import BeautifulSoup
import duckdb
import pandas as pd
from tqdm.asyncio import tqdm 
import numpy as np

conn = duckdb.connect('./data/duckdb.db')

cap_i = 1
cap_f = 950
max_concurrent_requests = 20
table_name = 'lord_of_mysteries'

semaphore = asyncio.Semaphore(max_concurrent_requests)

conn.execute(f'select max(capitulo) from {table_name}').df()

def chapter_diff(cap_i,cap_f,conn):

    capitulos_cadastrados = conn.execute(f'select distinct capitulo from {table_name}').df().iloc[:,0].to_numpy()
    capitulos_cadastrados.sort()
    lista_a_baixar = np.setdiff1d(np.arange(cap_i,cap_f+1),capitulos_cadastrados)

    return lista_a_baixar


async def scrap_chapter(session, n_cap):

    url = f'https://centralnovel.com/lord-of-mysteries-capitulo-{n_cap}/'
    try:
        async with session.get(url, timeout=10) as response:
            response.raise_for_status()
            html = await response.text()
            soup = BeautifulSoup(html, "html.parser")
            
            titulo = soup.find('div', {'class': "cat-series"}).text.strip()
            conteudo = soup.find('div',{'class':'epcontent entry-content'}).get_text(separator='\n',strip=True)

            return {
                'capitulo': n_cap,
                'titulo': titulo,
                'conteudo': conteudo
            }
    except Exception as e:
        return None


async def main():
    """
    Orquestra a execução de todas as tarefas assíncronas.
    """
    resultados = []
    capitulos_a_buscar = chapter_diff(cap_i,cap_f,conn)
    
    async with aiohttp.ClientSession() as session:
        tasks = [scrap_chapter(session, n) for n in capitulos_a_buscar]
        
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Baixando Capítulos (Async)"):
            resultado = await future
            if resultado:
                resultados.append(resultado)


    print(f"\nForam baixados {len(resultados)} capítulos com sucesso. Inserindo no banco de dados...")
    resultados.sort(key=lambda x: x['capitulo'])
    
    df = pd.DataFrame(resultados)

    # verificar se a tabela já foi criada
    if table_name not in conn.execute('show tables').fetch_df().values:
        print(f'Tabela {table_name} não encontrada no bando de dados')
        print(f'Criando tabela {table_name}')
        temp_df = df.head(1)
        conn.execute(f'create or replace table {table_name} as (select * from temp_df)')
        conn.execute(f'truncate table {table_name}')
        print(f'Tabela criada')
    
    conn.execute(f'insert into {table_name} (select * from df)')

    conn.commit()
    conn.close()

    # conn.execute(f'select count(1) from {table_name}').fetch_df()