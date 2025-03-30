import re
from collections import OrderedDict
import PyPDF2
import json

footer_patterns = [
    r'^CSV\s*:', 
    r'^DIRECCIÓN DE VALIDACIÓN\s*:', 
    r'^FIRMANTE\(1\)\s*:', 
    r'^\s*NOTAS\s*:',  # Just in case
]

def extract_pdf_structure(pdf_path):
    # Read PDF and extract text with page numbers
    pdf_file = open(pdf_path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    
    full_text = []
    for page_num, page in enumerate(pdf_reader.pages, start=1):
        text = page.extract_text()
        full_text.append(f"===== Page {page_num} =====\n{text}")
    
    # Extract first page as summary and remove it from further processing
    summary = full_text[0]
    content_pages = full_text[1:]
    
    # Process text to create structure
    structure = OrderedDict()
    chapter_summary = {}
    current_chapter = None
    current_chapter_num = None
    current_article = None
    current_page_start = None
    current_page_end = None
    page_number = 1  # starts from 2nd page now

    for page_content in content_pages:
        lines = page_content.split('\n')
        page_number += 1

        for i, line in enumerate(lines):
            # Skip common footers
            if any(re.match(pat, line) for pat in footer_patterns):
                continue

            # Detect new chapter line
            chapter_match = re.match(r'^CAPÍTULO ([IVXLCDM]+)\s*[-–]?\s*$', line.strip())
            if chapter_match:
                roman_num = chapter_match.group(1)
                current_chapter_num = roman_to_int(roman_num)

                # Try to get title from next line
                chapter_title = ""
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line:
                        chapter_title = next_line

                # Fallback if next line is empty
                if not chapter_title:
                    chapter_title = f"Sin título {roman_num}"

                current_chapter = f"CAPÍTULO {roman_num} - {chapter_title}"
                structure[current_chapter] = OrderedDict()
                chapter_summary[current_chapter_num] = {
                    "title": chapter_title,
                    "articles": []
                }
                continue
            
            # Detect new article
            article_match = re.match(r'Artículo (\d+)\.? (.*)', line)
            if article_match:
                art_num = int(article_match.group(1))
                art_title = article_match.group(2).strip()
                current_article = f"Artículo {art_num} - {art_title}"
                structure[current_chapter][current_article] = {
                    'text': [],
                    'pages': [page_number]
                }
                chapter_summary[current_chapter_num]['articles'].append(art_num)
                current_page_start = page_number
                current_page_end = page_number
                continue
            
            # Add content to current article
            if current_chapter and current_article in structure.get(current_chapter, {}):
                if line.strip():
                    structure[current_chapter][current_article]['text'].append(line)
                    if page_number != current_page_end:
                        current_page_end = page_number
                        structure[current_chapter][current_article]['pages'].append(page_number)
    
    # Clean up article texts and page ranges
    for chapter in structure:
        for article in structure[chapter]:
            # Combine text lines
            full_article_text = '\n'.join(structure[chapter][article]['text'])
            structure[chapter][article]['text'] = full_article_text

            # Simplify page ranges
            pages = structure[chapter][article]['pages']
            if len(pages) > 1:
                structure[chapter][article]['pages'] = f"{pages[0]}-{pages[-1]}"
            else:
                structure[chapter][article]['pages'] = str(pages[0])
    
    return structure, full_text, summary, chapter_summary

def roman_to_int(roman):
    roman_map = {'I': 1, 'V': 5, 'X': 10}
    result = 0
    prev = 0
    for char in reversed(roman):
        value = roman_map[char]
        if value < prev:
            result -= value
        else:
            result += value
        prev = value
    return result

# Use case
pdf_path = 'documents/ayudas_21-22.pdf'
document_structure, full_text, summary_text, chapter_summary = extract_pdf_structure(pdf_path)

# Save the structured text to .txt
with open('output.txt', 'w', encoding='utf-8') as f:
    f.write("=== SUMMARY ===\n")
    f.write(summary_text + "\n")
    f.write("=" * 80 + "\n\n")
    for chapter in document_structure:
        f.write(f"\n{chapter}\n")
        for article in document_structure[chapter]:
            f.write(f"  {article}\n")
            f.write(f"    Pages: {document_structure[chapter][article]['pages']}\n\n")
            f.write(f"{document_structure[chapter][article]['text']}\n")
            f.write("\n" + "=" * 80 + "\n\n")

# Print structure summary
# for chapter in document_structure:
#     print(f"\n{chapter}")
#     for article in document_structure[chapter]:
#         print(f"  {article}")
#         print(f"    Pages: {document_structure[chapter][article]['pages']}")
#         print(f"    Text sample: {document_structure[chapter][article]['text'][:100]}...")

# Save structure to JSON
# with open('structure.json', 'w', encoding='utf-8') as json_file:
#     json.dump(document_structure, json_file, ensure_ascii=False, indent=2)

# Save chapter summary index to JSON
with open('structure_index.json', 'w', encoding='utf-8') as index_file:
    json.dump(chapter_summary, index_file, ensure_ascii=False, indent=2)
