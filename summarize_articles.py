import re
import fitz # PyMuPDF
import json
import os
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer

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

# Function to summarize text using Mixtral 8x7B
def summarize_with_mixtral(text, model, tokenizer):
    """
    Generate a summary of the text using Mixtral 8x7B.
    
    Args:
        text (str): The text to summarize
        model: The Mixtral model
        tokenizer: The tokenizer for the model
        
    Returns:
        str: The generated summary
    """
    # Create a prompt that instructs the model to summarize the text in Spanish
    prompt = f"Resume el siguiente texto de forma concisa en español:\n\n{text}\n\nResumen:"
    
    # Tokenize the input text
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    
    # Generate the summary
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=150,  # Limit the length of the summary
        temperature=0.7,     # Add some randomness to generation
        do_sample=True
    )
    
    # Decode the generated tokens into text
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the summary part (after "Resumen:")
    if "Resumen:" in summary:
        summary = summary.split("Resumen:")[1].strip()
        
    return summary

# Sample usage
if __name__ == "__main__":
    # Initialize Mixtral 8x7B model
    print("Loading Mixtral 8x7B model...")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
    
    pdf_files = ["documents/ayudas_23-24.pdf"]
    print(f"Processing {pdf_files[0]}...")
    document_text = extract_text_from_pdf2(pdf_files[0])
    
    file = "texto"
    with open(f"{file}.txt", "w") as file:
        # Write the string to the file
        file.write(document_text)
    
    cleaned_text = remove_pattern(document_text)
    articles = parse_articles(cleaned_text)
    
    # Apply Mixtral 8x7B to summarize each article
    print("Summarizing articles...")
    articles_with_summaries = {}
    
    for key, value in articles.items():
        print(f"Summarizing {key}...")
        # For each article, generate a summary
        summary = summarize_with_mixtral(value, model, tokenizer)
        
        # Replace the original text with a dictionary containing both original text and summary
        articles[key] = {
            "original_text": value,
            "summary": summary
        }
    
    # Save the updated articles dictionary to a JSON file
    with open("articles.json", "w") as file:
        json.dump(articles, file, indent=2)
    
    # Print some examples
    print("\nResults:")
    for key, value in list(articles.items())[:2]:  # Show just first 2 examples
        print(f"\n{key}")
        print(f"Original: {value['original_text'][:100]}...")
        print(f"Summary: {value['summary']}")
    
    print("\nAll articles have been processed and saved to articles.json")