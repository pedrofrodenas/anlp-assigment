import re
import os
import fitz  # PyMuPDF
import json

def extract_text_from_pdf2(pdf_path):
    """Extracts text from each page of a PDF file, returning a list of pages with their numbers and text."""
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):  # Page numbers start at 1
        text = page.get_text()
        pages.append({'page_num': page_num, 'text': text})
    return pages

def remove_pattern(text):
    pattern = r'Código seguro de Verificación : GEN-afed-f11a-2acf-695d-4be9-51c8-2c08-5869 \| Puede verificar la integridad de este documento en la siguiente dirección : https://sede\.administracion\.gob\.es/pagSedeFront/servicios/consult\.\.\.\nCSV : GEN-afed-f11a-2acf-695d-4be9-51c8-2c08-5869\nDIRECCIÓN DE VALIDACIÓN : https://sede\.administracion\.gob\.es/pagSedeFront/servicios/consultaCSV\.htm\nFIRMANTE\(1\) : JOSE MANUEL BAR CENDÓN \| FECHA : 15/03/2023 16:43 \| Aprueba\n'
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

def parse_articles(combined_text, page_ranges):
    """Parse articles from combined text and determine the pages each article spans."""
    pattern = re.compile(
        r'(Artículo \d+\.\s*.*?)(?=\s*(?:Artículo \d+\.|CAPÍTULO|SECCIÓN|\Z))',
        re.DOTALL
    )
    articles_dict = {}
    
    for match in pattern.finditer(combined_text):
        article_text = match.group(1)
        start_pos = match.start()
        end_pos = match.end()
        
        # Determine which pages this article spans
        pages = set()
        for (page_start, page_end, page_num) in page_ranges:
            if start_pos < page_end and end_pos > page_start:
                pages.add(page_num)
        
        # Split into title and body
        title_end = article_text.find('\n')
        if title_end == -1:
            title = article_text.strip()
            body = ""
        else:
            title = article_text[:title_end].strip()
            body = article_text[title_end+1:].strip()
        
        # Extract article number
        match_num = re.search(r'Artículo (\d+)\.', title)
        if match_num:
            num = int(match_num.group(1))
            articles_dict[num] = {
                'title': title,
                'content': body,
                'pages': sorted(pages)  # Store sorted list of pages
            }
    
    return articles_dict

# Sample usage
if __name__ == "__main__":
    pdf_files = ["documents/ayudas_21-22.pdf", 
                 "documents/ayudas_22-23.pdf",
                 "documents/ayudas_23-24.pdf", 
                 "documents/ayudas_24-25.pdf",
                 "documents/ayudas_25-26.pdf"]

    # Create 'text' directory if it doesn't exist
    os.makedirs("text", exist_ok=True)

    main_dict = {}

    for pdf_path in pdf_files:
        print(f"Processing {pdf_path}...")
        # Extract pages with text and page numbers
        pages = extract_text_from_pdf2(pdf_path)
        # Clean each page's text
        cleaned_pages = []
        for page in pages:
            cleaned_text = remove_pattern(page['text'])
            cleaned_pages.append({'page_num': page['page_num'], 'text': cleaned_text})
        # Build combined_text and page_ranges
        combined_text = ''
        page_ranges = []
        for page in cleaned_pages:
            start = len(combined_text)
            combined_text += page['text']
            end = len(combined_text)
            page_ranges.append( (start, end, page['page_num']) )
        # Save cleaned text to a text file
        base_name = os.path.basename(pdf_path)
        txt_filename = os.path.splitext(base_name)[0] + ".txt"
        txt_path = os.path.join("text", txt_filename)
        with open(txt_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(combined_text)
        # Parse articles and add to main dictionary
        articles = parse_articles(combined_text, page_ranges)
        main_dict[base_name] = articles

    # Save the main dictionary to JSON
    with open("documents_parsed.json", "w", encoding="utf-8") as file:
        json.dump(main_dict, file, indent=2, ensure_ascii=False)