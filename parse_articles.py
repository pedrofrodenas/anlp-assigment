import re
import fitz  # PyMuPDF
import re
import json
import os # Added for file path handling


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
        articles_dict[title] = body
    return articles_dict

# Function to process the PDF text from the document
def process_pdf_text(document_input):
    """
    Process the PDF text and extract all articles
    
    Args:
        document_input (str): The raw document input containing the PDF text
        
    Returns:
        dict: Dictionary of extracted articles
    """
    # Combine all pages into a single text
    full_text = ""
    
    # Extract content from each page
    page_pattern = re.compile(r'<document_content page="\d+">(.*?)</document_content>', re.DOTALL)
    for match in page_pattern.finditer(document_input):
        full_text += match.group(1) + "\n"
    
    # Extract articles from the combined text
    return extract_articles(full_text)

# Sample usage
if __name__ == "__main__":

    pdf_files = ["documents/ayudas_23-24.pdf"]

    print(f"Processing {pdf_files[0]}...")
    document_text = extract_text_from_pdf2(pdf_files[0])

    file= "texto"

    with open(f"{file}.txt", "w") as file:
        # Write the string to the file
        file.write(document_text)

    
    cleaned_text = remove_pattern(document_text)

    articles = parse_articles(cleaned_text)

    json_string = json.dumps(articles, indent=2)

    with open("articles.json", "w") as file:
        json.dump(articles, file, indent=2)

    

    for key, value in articles.items():
        print(key)
        print(value)

    

    # Assuming document_input contains the raw PDF text
    # articles = process_pdf_text(document_input)
    
    # For demonstration, print what the output would look like
    print("Example output would show something like:")
    print("{'Artículo 1. Objeto y beneficiarios': '1. Se convocan por la presente Resolución becas para estudiantes que en el curso académico 2023-2024, cursen enseñanzas postobligatorias con validez en todo el territorio nacional...'}")
    print("{'Artículo 2. Financiación de la convocatoria': '1. El importe correspondiente a las becas que se convocan se hará efectivo con cargo al crédito 18.08.323M.482.00 de los presupuestos del Ministerio de Educación y Formación Profesional...'}")