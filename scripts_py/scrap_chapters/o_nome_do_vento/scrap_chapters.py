import os
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup



book = epub.read_epub('./scripts_py/scrap_chapters/o_nome_do_vento/O Nome do Vento - A Cronica do Matador do Rei - Vol.  1  - Patrick Rothfuss.epub')


toc_items = book.toc

item = toc_items[7]



href_item = toc_items[7].href

content_item = book.get_item_with_href(href_item)


soup = BeautifulSoup(content_item.get_content(), 'html.parser')


text = soup.get_text(separator='\n',strip=True)



with open('chapter.txt','w') as file:
    file.write(text)