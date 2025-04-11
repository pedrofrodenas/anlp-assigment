import json
import os
import re
from topics import topics
from tqdm import tqdm
import torch as th
import warnings
warnings.filterwarnings("ignore")


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


def create_markdown_file(document_name, organized_data, topk=None):
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
                    for ith, article in enumerate(sorted_articles if topk is None else sorted_articles[:topk]):
                        # combined_text += f"\n### TITLE: {article['title']}\n{article['summary']}\n"
                        if ith != 0 and article['summary'].strip()[0] in ['1', '-']:
                            combined_text += f"\n#### {article['title']}\n{article['summary']}\n"  # WHICH WORKS BETTER? TODO
                        else:
                            combined_text += f"\n{article['summary']}\n"  # WHICH WORKS BETTER? TODO
                    
                    # Prepare references for all articles
                    references = []
                    for article in sorted_articles:
                        article_title = article["title"].strip()
                        if len(article['pages']) > 1:
                            pages = f'{min(article["pages"])}-{max(article["pages"])}'
                        else:
                            pages = f'{min(article["pages"])}'

                        references.append((article['id'], f"Art. {article['id']}: {article_title} (pp. {pages})"))
                    
                    if combined_text:
                        # Normalize the new lines format
                        combined_text = [line.strip() for line in combined_text.split('\n')]
                        combined_text = "\n".join(combined_text)

                        md_file.write(f"{combined_text}\n\n")
                        md_file.write("**Artículos relacionados:**\n\n")
                        for _, ref in sorted(references, key=lambda x: int(x[0])):
                            md_file.write(f"- {ref}\n")
                        md_file.write("\n")
    
    print(f"Created markdown file: {output_filename}")
    return output_filename

def process_documents(json_data_path, topk=None):
    # Load the data
    summarized_data = load_summarized_data(json_data_path)

    # Process each document
    for document_name, document_data in summarized_data.items():
        print(f"Processing {document_name}...")
        organized_data = group_articles_by_topic(document_data)
        output_md_file = create_markdown_file(document_name, organized_data, topk=topk)
        
        try:  # Save as pdf if md2pdf exists
            from md2pdf.core import md2pdf
            md2pdf(output_md_file.replace('.md', '.pdf'), md_file_path=output_md_file)
        except ModuleNotFoundError:
            print('Could not convert markdown to pdf as md2pdf is not installed')


if __name__ == "__main__":
    json_data_path = "documents_summarized_with_topics.json"
    process_documents(json_data_path, topk=3)