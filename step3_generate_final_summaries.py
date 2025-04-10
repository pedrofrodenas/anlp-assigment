import json
import os
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from topics import topics
from tqdm import tqdm
import torch as th
import warnings
warnings.filterwarnings("ignore")

# Helper function to clean up whitespace
WHITESPACE_HANDLER = lambda k: re.sub(r'\s+', ' ', re.sub(r'\n+', ' ', k.strip()))
MODEL = "meta-llama/Llama-3.2-1B-Instruct"  # Changed to Llama 3.2

# Set the device for PyTorch, if GPU is available and then if MPS is available
if th.cuda.is_available():
    device = th.device("cuda")
elif th.backends.mps.is_available():
    device = th.device("mps")
else: 
    device = th.device("cpu")
print(f"Using device: {device}")


def load_summarized_data(file_path):
    """Load the summarized articles with topics"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def initialize_topic_structure():
    """Create a nested dictionary based on topics structure"""
    topic_dict = {}
    for topic, subtopic in topics:
        if topic not in topic_dict:
            topic_dict[topic] = {}
        topic_dict[topic][subtopic] = []
    return topic_dict

def group_articles_by_topic(document_data):
    """Group all articles by their assigned topic and subtopic"""
    organized_data = initialize_topic_structure()
    
    # Iterate through all articles in the document
    for article_id, article_data in document_data.items():
        if isinstance(article_data, dict) and "topic" in article_data and "subtopic" in article_data:
            topic = article_data["topic"]
            subtopic = article_data["subtopic"]
            
            # Check if this topic and subtopic combination exists in our structure
            if topic in organized_data and subtopic in organized_data[topic]:
                # Add the article to the appropriate topic/subtopic
                organized_data[topic][subtopic].append({
                    "id": article_id,
                    "title": article_data.get("title-trimmed", ""),
                    "summary": article_data.get("summary", ""),
                    "pages": article_data.get("pages", []),
                    "score": article_data.get("topic_score", 0)
                })
    
    return organized_data

def generate_summary(text_to_summarize, tokenizer, model):
    """Generate a summary for the given text using the model"""

    # Create a specialized prompt with explicit end marker
    prompt = f"""Eres un asistente legal especializado en resumir información sobre becas y ayudas para estudiantes españoles. 
                Debes resumir el siguiente texto legal relacionado con becas y ayudas al estudio. 
                Enfócate en extraer la información más valiosa para un estudiante que quiere solicitar una beca.

                Proporciona un resumen conciso pero informativo que permita a los estudiantes entender los aspectos 
                más importantes para acceder a estas becas. Omite detalles técnicos innecesarios.
                
                El resumen debe tener un máximo de 300 palabras y me gustaria que lo estructuraras como un filete de formato markdown.

                TEXTO PARA RESUMIR:
                {text_to_summarize}

                RESUMEN:
                """

    # Tokenize the input text
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Calculate available tokens for generation
    num_input_tokens = inputs["input_ids"].shape[1]

    # Limit summary to around 300-400 words (~600 tokens)
    max_new_tokens = min(600, 5020 - num_input_tokens)

    # Generate summary with more controlled parameters
    with th.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=100,  # Ensure at least some summary content
            num_beams=4,
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
            no_repeat_ngram_size=2,
            early_stopping=True,  # Stop when a stopping criteria is met
            length_penalty=2.0,   # Penalize lengthy generations
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode the generated tokens
    summary = WHITESPACE_HANDLER(tokenizer.decode(outputs[0], skip_special_tokens=True))

    # Extract just the summary part (after "RESUMEN:")
    if "RESUMEN:" in summary:
        summary = summary.split("RESUMEN:")[1].strip()

    return summary

def create_markdown_file(document_name, organized_data, tokenizer, model):
    """Create a markdown file with summaries organized by topic and subtopic"""
    # Ensure the output directory exists
    os.makedirs("resumen_por_año", exist_ok=True)
    
    output_filename = f"resumen_por_año/{document_name.split('.')[0]}_summary.md"
    
    with open(output_filename, 'w', encoding='utf-8') as md_file:
        # Write document title
        doc_title = document_name.replace('.pdf', '').replace('_', ' ').title()
        md_file.write(f"# Resumen de {doc_title}\n\n")
        
        # Create table of contents
        md_file.write("## Índice\n\n")
        for topic, subtopics in organized_data.items():
            has_content = any(organized_data[topic][subtopic] for subtopic in subtopics)
            if has_content:
                md_file.write(f"- [{topic}](#{topic.lower().replace(' ', '-').replace(',', '').replace('/', '-')})\n")
                for subtopic in subtopics:
                    if organized_data[topic][subtopic]:
                        md_file.write(f"  - [{subtopic}](#{subtopic.lower().replace(' ', '-').replace(',', '').replace('/', '-')})\n")
        md_file.write("\n---\n\n")
        
        # Write content for each topic and subtopic using tqdm to visualize progress
        for topic, subtopics in tqdm(organized_data.items(), desc="Processing topics"):
            topic_has_content = any(organized_data[topic][subtopic] for subtopic in subtopics)
            
            if topic_has_content:
                md_file.write(f"## {topic}\n\n")
                
                for subtopic, articles in tqdm(subtopics.items(), desc=f"Processing '{topic}' subtopics", leave=False):
                    if not articles:
                        continue
                        
                    md_file.write(f"### {subtopic}\n\n")
                    
                    # Sort articles by score (highest first)
                    sorted_articles = sorted(articles, key=lambda x: x["score"], reverse=True)
                    
                    # Combine summaries from all articles
                    combined_text = ""
                    for article in sorted_articles:
                        combined_text += f"{article['title']}\n{article['summary']}"
                        # combined_text += f"{article['summary']} "  # WHICH WORKS BETTER? TODO
                    
                    # Prepare references for all articles
                    references = []
                    for article in sorted_articles:
                        article_title = article["title"].strip()
                        if len(article['pages']) > 1:
                            pages = f'{min(article["pages"])}-{max(article["pages"])}'
                        else:
                            pages = f'{min(article["pages"])}'

                        references.append(f"Art. {article['id']}: {article_title} (pp. {pages})")
                    
                    if combined_text:
                        final_summary = generate_summary(combined_text, tokenizer, model)
                        md_file.write(f"{final_summary}\n\n")
                        md_file.write("**Artículos relacionados:**\n\n")
                        for ref in references:
                            md_file.write(f"- {ref}\n")
                        md_file.write("\n")
    
    print(f"Created markdown file: {output_filename}")
    return output_filename

def process_documents(json_data_path):
    """Process all documents and generate summaries"""
    # Load the model and tokenizer
    print("Loading model and tokenizer...")

    # Load tokenizer - no fast tokenizer for Llama models
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model without quantization
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        device_map="auto",
        torch_dtype=th.float16
    )
    print("Model loaded")

    # Load the data
    summarized_data = load_summarized_data(json_data_path)

    # Process each document
    for document_name, document_data in summarized_data.items():
        print(f"Processing {document_name}...")
        organized_data = group_articles_by_topic(document_data)
        output_md_file = create_markdown_file(document_name, organized_data, tokenizer, model)
        
        try:  # Save as pdf if md2pdf exists
            from md2pdf.core import md2pdf
            md2pdf(output_md_file.replace('.md', '.pdf'), md_file_path=output_md_file)
        except ModuleNotFoundError:
            print('Could not convert markdown to pdf as md2pdf is not installed')
        
        print('generating only first for debugging purposes')
        break

if __name__ == "__main__":
    json_data_path = "documents_summarized_with_topics.json"
    process_documents(json_data_path)
