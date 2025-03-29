import re
import fitz  # PyMuPDF
import json

def extract_text_from_pdf2(pdf_path):
    """Extracts text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def remove_pattern(text):
    pattern = r'Código seguro de Verificación : GEN-afed-f11a-2acf-695d-4be9-51c8-2c08-5869 \| Puede verificar la integridad de este documento en la siguiente dirección : https://sede\.administracion\.gob\.es/pagSedeFront/servicios/consult\.\.\.\nCSV : GEN-afed-f11a-2acf-695d-4be9-51c8-2c08-5869\nDIRECCIÓN DE VALIDACIÓN : https://sede\.administracion\.gob\.es/pagSedeFront/servicios/consultaCSV\.htm\nFIRMANTE\(1\) : JOSE MANUEL BAR CENDÓN \| FECHA : 15/03/2023 16:43 \| Aprueba\n'
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

def parse_articles(text):
    # Regex pattern to match articles, using lookahead for next article or section
    pattern = re.compile(
        r'(Artículo \d+\.\s*.*?)(?=\s*(?:Artículo \d+\.|CAPÍTULO|SECCIÓN|\Z))',
        re.DOTALL
    )
    articles = pattern.findall(text)
    
    articles_dict = {}
    for article in articles:
        # Split into title and body
        title_end = article.find('\n')
        if title_end == -1:
            title = article.strip()
            body = ""
        else:
            title = article[:title_end].strip()
            body = article[title_end+1:].strip()
        # Extract article number
        match = re.search(r'Artículo (\d+)\.', title)
        if match:
            num = int(match.group(1))
            articles_dict[num] = {
                'title': title,
                'content': body
            }
    return articles_dict

# Sample usage
if __name__ == "__main__":
    pdf_files = ["documents/ayudas_23-24.pdf"]

    print(f"Processing {pdf_files[0]}...")
    document_text = extract_text_from_pdf2(pdf_files[0])

    cleaned_text = remove_pattern(document_text)

    articles = parse_articles(cleaned_text)

    with open("articles.json", "w") as file:
        json.dump(articles, file, indent=2, ensure_ascii=False)
