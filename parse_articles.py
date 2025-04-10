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
    # Combined regex pattern to handle all verification block variations
    pattern = r'''(?mx)
    (
        # Pattern 1: Name - Date line with CSV URL
        ^.*?-\s+\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+CET\s*\n
        Puede\s+comprobar.*?https://\S+
    )
    |
    (
        # Pattern 2: Multi-line verification blocks
        ^(?:Código\s+seguro\s+de\s+Verificación\s*:|CSV\s*:).*?
        (?:\n(?:CSV\s*:|DIRECCIÓN\s+DE\s+VALIDACIÓN\s*:|Puede\s+verificar|FIRMANTE\(1\)\s*:).*?)*
        \nFIRMANTE\(1\)\s*:.*
    )
    '''
    
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

def parse_articles(combined_text, page_ranges):
    """Parse articles and chapters from combined text and determine the pages each article spans."""
    # Pattern to find chapters first
    chapter_pattern = re.compile(
        r'CAPÍTULO\s+([IVXLCDM]+)\s*\n(.*?)\s*(?=\s*(?:Artículo \d+\.|CAPÍTULO|SECCIÓN|\Z))',
        re.DOTALL
    )
    chapters = []
    for match in chapter_pattern.finditer(combined_text):
        start = match.start()
        end = match.end()
        num = match.group(1).strip()
        name = match.group(2).strip()
        chapters.append({'start': start, 'end': end, 'number': num, 'name': name})
    
    # Pattern to find articles
    article_pattern = re.compile(
        r'(Artículo \d+\.\s*.*?)(?=\s*(?:Artículo \d+\.|CAPÍTULO|SECCIÓN|\Z))',
        re.DOTALL
    )
    articles_dict = {}
    
    for match in article_pattern.finditer(combined_text):
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
        
        # Extract article number and trim title
        match_num = re.search(r'Artículo (\d+)\.', title)
        if match_num:
            num = int(match_num.group(1))
            trimmed_title = re.sub(r'^Artículo \d+\.\s*', '', title).strip()
            
            # Determine current chapter for this article
            current_chapter = None
            for chap in chapters:
                if chap['start'] <= start_pos:
                    current_chapter = chap
                else:
                    break
            
            # Prepare article info
            article_info = {
                'title': title,
                'title-trimmed': trimmed_title,
                'content': body,
                'pages': sorted(pages)
            }
            
            # Add chapter info if available
            if current_chapter:
                article_info['chapter_number'] = current_chapter['number']
                article_info['chapter_name'] = current_chapter['name']
            
            articles_dict[num] = article_info
    
    return articles_dict

# Sample usage remains unchanged
if __name__ == "__main__":
    pdf_files = ["documents/ayudas_21-22.pdf", 
                 "documents/ayudas_22-23.pdf",
                 "documents/ayudas_23-24.pdf", 
                 "documents/ayudas_24-25.pdf",
                 "documents/ayudas_25-26.pdf"]

    os.makedirs("text", exist_ok=True)

    main_dict = {}

    for pdf_path in pdf_files:
        print(f"Processing {pdf_path}...")
        pages = extract_text_from_pdf2(pdf_path)
        cleaned_pages = []
        for page in pages:
            cleaned_text = remove_pattern(page['text'])
            cleaned_pages.append({'page_num': page['page_num'], 'text': cleaned_text})
        combined_text = ''
        page_ranges = []
        for page in cleaned_pages:
            start = len(combined_text)
            combined_text += page['text']
            end = len(combined_text)
            page_ranges.append((start, end, page['page_num']))
        base_name = os.path.basename(pdf_path)
        txt_filename = os.path.splitext(base_name)[0] + ".txt"
        txt_path = os.path.join("text", txt_filename)
        with open(txt_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(combined_text)
        articles = parse_articles(combined_text, page_ranges)
        main_dict[base_name] = articles

    with open("documents_parsed.json", "w", encoding="utf-8") as file:
        json.dump(main_dict, file, indent=2, ensure_ascii=False)