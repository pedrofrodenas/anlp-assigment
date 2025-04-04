import re
#import fitz  # PyMuPDF
import json
import os
import requests
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from huggingface_hub import login

login()


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


def roman_to_int(roman):
    roman_map = {'I': 1, 'V': 5, 'X': 10, 'L': 50}
    result, prev = 0, 0
    for char in reversed(roman):
        val = roman_map[char]
        result = result - val if val < prev else result + val
        prev = val
    return result


def parse_articles_grouped_by_chapter(combined_text, page_ranges):
    """
    Parse chapters and articles from the combined text.
    Returns structure:
    { chapter_num: { title: ..., articles: { article_num: {...} } } }
    """
    pattern = re.compile(
        r'(CAP[IÍ]TULO\s+[IVXLCDM]+.*?(?:\n.+)?\n)?(Artículo \d+\.\s*.*?)(?=\s*(?:Artículo \d+\.|CAP[IÍ]TULO|SECCIÓN|\Z))',
        re.DOTALL
    )

    data = {}
    current_chapter_num = "0"
    current_chapter_title = "Sin título"

    for match in pattern.finditer(combined_text):
        chapter_header, article_text = match.groups()

        # Detect new chapter
        if chapter_header:
            roman_match = re.search(r'CAP[IÍ]TULO\s+([IVXLCDM]+)', chapter_header)
            title_match = re.findall(r'\n(.+)', chapter_header.strip())

            if roman_match:
                current_chapter_num = str(roman_to_int(roman_match.group(1)))

            current_chapter_title = title_match[0].strip() if title_match else f"Sin título {current_chapter_num}"

            if current_chapter_num not in data:
                data[current_chapter_num] = {
                    "title": current_chapter_title,
                    "articles": {}
                }

        # Determine pages
        start_pos = match.start()
        end_pos = match.end()
        pages = set()
        for (page_start, page_end, page_num) in page_ranges:
            if start_pos < page_end and end_pos > page_start:
                pages.add(page_num)

        # Extract article details
        title_end = article_text.find('\n')
        title = article_text.strip() if title_end == -1 else article_text[:title_end].strip()
        body = "" if title_end == -1 else article_text[title_end + 1:].strip()
        match_num = re.search(r'Artículo (\d+)\.', title)
        if not match_num:
            continue
        article_id = match_num.group(1)
        trimmed_title = re.sub(r'^Artículo \d+\.\s*', '', title).strip()

        # Ensure current chapter is initialized
        if current_chapter_num not in data:
            data[current_chapter_num] = {
                "title": current_chapter_title,
                "articles": {}
            }

        data[current_chapter_num]["articles"][article_id] = {
            "title": title,
            "title-trimmed": trimmed_title,
            "content": body,
            "pages": sorted(pages)
        }

    return data


def summarize_with_mistral(text, model, tokenizer):
    """Generate a summary using the Mistral-Nemo-Instruct model."""
    # Format the instruction prompt for Mistral-Nemo-Instruct
    prompt = f"<|im_start|>user\nResume el siguiente texto con tus propias palabras de forma muy breve, evitando copiar frases textuales :\n\n{text}<|im_end|>\n<|im_start|>assistant\n"
    prompt = f"<|im_start|>user\nTLDR de este texto en español, sé extremadamente breve y directo:\n\n{text}<|im_end|>\n<|im_start|>assistant\n"
    # prompt = f"<|im_start|>user\nCrea un resumen ultra conciso del siguiente texto en un máximo de 50 palabras, captando solo los puntos más esenciales:\n\n{text}<|im_end|>\n<|im_start|>assistant\n"
    prompt = f"<|im_start|>user\nProporciona un resumen ultra-breve con esta estructura exacta: 'Objetivo: [una frase]. Alcance: [una frase]. Conclusión: [una frase]'.\n\n{text}<|im_end|>\n<|im_start|>assistant\n"
    prompt = f"<|im_start|>user\nExtrae solo los 3-5 conceptos más importantes de este texto, sin elaboraciones innecesarias:\n\n{text}<|im_end|>\n<|im_start|>assistant\n"
    prompt = f"<|im_start|>user\nResume este artículo en formato: (1) Contexto general en una frase, (2) 3-5 conceptos clave sin elaboraciones innecesarias:\n\n{text}<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Set pad_token_id if it's None
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.7,
        do_sample=True,
        early_stopping=True, 
        num_return_sequences=1,
        num_beams=4, 
        pad_token_id=pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    # Extract the summary from the generated text
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Improved cleanup logic to remove all special tokens
    # First, try to extract just the assistant's response
    if "<|im_start|>assistant\n" in summary:
        summary = summary.split("<|im_start|>assistant\n")[1].strip()
    
    # Then, make sure to remove any trailing <|im_end|> token
    if "<|im_end|>" in summary:
        summary = summary.split("<|im_end|>")[0].strip()

    print(summary)
    
    return summary


if __name__ == "__main__":
    print("Loading Mistral-Nemo-Instruct-2407 model...")
    
    # Load the Mistral-Nemo-Instruct model
    model_name = "gsarti/it5-small-wiki-summarization"
    # model_name = "mistralai/Ministral-8B-Instruct-2410"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16  # Use float16 for efficiency
    )

    pdf_files = [
        "documents/ayudas_21-22.pdf",
        "documents/ayudas_22-23.pdf",
        "documents/ayudas_23-24.pdf",
        "documents/ayudas_24-25.pdf",
        "documents/ayudas_25-26.pdf"
    ]

    os.makedirs("text", exist_ok=True)
    # main_dict = {}

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
        # Parse articles
        #articles = parse_articles(combined_text, page_ranges)
        articles_by_chapter = parse_articles_grouped_by_chapter(combined_text, page_ranges)

        
        print(f"Summarizing articles in {base_name}...")
        # for article_num, article_data in articles.items():
        #     print(f"Summarizing Article {article_num}...")
        #     summary = summarize_with_mistral(article_data['content'], model, tokenizer)
        #     article_data['summary'] = summary
        main_dict[base_name] = {}

        for chapter_num, chapter_data in articles_by_chapter.items():
            chapter_title = chapter_data["title"]
            article_map = chapter_data["articles"]

            main_dict[base_name][chapter_num] = {
                "title": chapter_title,
                "articles": {}
            }

            for article_num, article_data in article_map.items():
                print(f"Summarizing Article {article_num} in Chapter {chapter_num}...")
                summary = summarize_with_mistral(article_data['content'], model, tokenizer)
                article_data['summary'] = summary
                main_dict[base_name][chapter_num]["articles"][article_num] = article_data


    with open("documents_summarized.json", "w", encoding="utf-8") as file:
        json.dump(main_dict, file, indent=2, ensure_ascii=False)

    print("\nAll documents have been processed and saved to documents_summarized.json")
