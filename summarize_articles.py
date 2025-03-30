import re
import fitz  # PyMuPDF
import json
import os
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from huggingface_hub import login

login()

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
    pattern = re.compile(
        r'(Artículo \d+\.\s*.*?)(?=\s*(?:Artículo \d+\.|CAPÍTULO|SECCIÓN|\Z))',
        re.DOTALL
    )
    articles = pattern.findall(text)
    
    articles_dict = {}
    for article in articles:
        title_end = article.find('\n')
        if title_end == -1:
            title = article.strip()
            body = ""
        else:
            title = article[:title_end].strip()
            body = article[title_end+1:].strip()
        match = re.search(r'Artículo (\d+)\.', title)
        if match:
            num = int(match.group(1))
            articles_dict[num] = {
                'title': title,
                'content': body
            }
    return articles_dict

def summarize_with_mixtral(text, model, tokenizer):
    prompt = f"Resume el siguiente texto de forma concisa en español:\n\n{text}\n\nResumen:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=150,
        temperature=0.7,
        do_sample=True
    )
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Resumen:" in summary:
        summary = summary.split("Resumen:")[1].strip()
    return summary

if __name__ == "__main__":
    print("Loading Mixtral 8x7B model...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mixtral-8x7B-v0.1",
        device_map="auto",
        quantization_config=quantization_config
    )

    pdf_files = [
        "documents/ayudas_21-22.pdf",
        "documents/ayudas_22-23.pdf",
        "documents/ayudas_23-24.pdf",
        "documents/ayudas_24-25.pdf",
        "documents/ayudas_25-26.pdf"
    ]

    os.makedirs("text", exist_ok=True)
    main_dict = {}

    for pdf_path in pdf_files:
        print(f"Processing {pdf_path}...")
        document_text = extract_text_from_pdf2(pdf_path)
        cleaned_text = remove_pattern(document_text)
        
        base_name = os.path.basename(pdf_path)
        txt_filename = os.path.splitext(base_name)[0] + ".txt"
        txt_path = os.path.join("text", txt_filename)
        with open(txt_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(cleaned_text)
        
        articles = parse_articles(cleaned_text)
        
        print(f"Summarizing articles in {base_name}...")
        for article_num, article_data in articles.items():
            print(f"Summarizing Article {article_num}...")
            summary = summarize_with_mixtral(article_data['content'], model, tokenizer)
            article_data['summary'] = summary
        
        main_dict[base_name] = articles

    with open("documents_summarized.json", "w", encoding="utf-8") as file:
        json.dump(main_dict, file, indent=2, ensure_ascii=False)


    print("\nAll documents have been processed and saved to documents_parsed.json")