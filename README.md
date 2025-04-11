# Anlp-Assigment

This project aims to extract, summarize, and organize information from Spanish legal PDFs, specifically focusing on scholarship and grant announcements. The goal is to create a structured and easily searchable resource for students and researchers.

## Project Structure

The project is structured to handle the complexities of Spanish legal documents, which often have a hierarchical organization.  

- **Level 1: Resoluciones (Resolutions)** - These are the main documents outlining the scholarship/grant program.  
- **Level 2: Capítulos (Chapters)** - These divide the resolution into thematic sections.  
- **Level 3: Artículos (Articles)** - These are the individual clauses or regulations within each chapter.

## Key Features

- **PDF Extraction:** Uses PyMuPDF to extract text from Spanish legal PDFs.  
- **Article Parsing:** Parses individual articles and tracks their page locations within the original PDF.  
- **Summarization:** Generates concise summaries of articles using the Mistral-Nemo-Instruct LLM.  
- **Structured JSON Output:** Saves both raw text and summaries to a structured JSON format for easy processing and searching.  
- **Topic Classification:** Uses BERT embeddings to classify articles into predefined topics and subtopics.  
- **Abstract Summary Generation:** Creates a high-level abstract summary by identifying and summarizing relevant articles, with page references for non-relevant ones.

## Workflow

1.**PDF Extraction:** The script extracts text from the PDF using PyMuPDF.  
2.**Article Parsing:** The extracted text is parsed to identify individual articles and their corresponding page numbers.  
3.**Summarization:** Each article is summarized using the Mistral-Nemo-Instruct LLM.  
4.**Topic Classification:** BERT embeddings are used to classify articles into predefined topics and subtopics.  

- Topic Text: "TOPICNAME - Brief Description"  
- Pdf extracted Text: "CHAPTERNAME"  
- Tokenize(T) -> (N,)  
- Tokenize(Q) -> (M,)  
- T\_bert: Bert(Tokenize(T)) -> (N, E)  
- Q\_bert: Bert(Tokenize(Q)) -> (M, E)  
- Compute mean over token dimension  
- Mean(T\_bert) -> (1, E)  
- Mean(Q\_bert) -> (1, E)  

5.**Abstract Summary Generation:**  The script identifies and summarizes relevant articles for the abstract summary, including page references for non-relevant articles.  
6.**JSON Output:** The extracted data, summaries, and classifications are saved to a structured JSON file.

## File Structure

The project is organized as follows:

- **`README.md`**: This file, providing an overview of the project.  
- **`main.py`**: The main script that orchestrates the PDF processing and summarization.  
- **`documents/`**: A directory containing the PDF documents to be processed.  (Example files: `ayudas_21-22.pdf`, `ayudas_22-23.pdf`, etc.)  
- **`text/`**: A directory where the extracted text from PDFs is stored as `.txt` files. These files are intermediate results and are automatically generated during the processing.  
- **`documents_summarized.json`**: The output file containing the summarized articles in JSON format. This file is generated after the script has finished processing all PDF documents.  

## Running the Code

To run the script, ensure you have Python 2.7 or 3.x installed along with the required libraries.  You can install the necessary packages using pip:

```bash
pip install PyPDF2 transformers torch accelerate bitsandbytes
```

After installing the dependencies, navigate to the root directory of the repository in your terminal and execute the following command:

```bash
python main.py
```

The script will process each PDF file in the documents/ directory, extract the text, parse the articles, generate summaries using the Mistral-Nemo-Instruct model, and save the results to documents_summarized.json.

## Example Topics and Subtopics

- Requisitos y Solicitud
- Criterios de Selección
- Convocatoria y Plazos
- Matrícula y Expedientes
- Control, Verificación y Reintegro
- Compatibilidades e Incompatibilidades con otras ayudas
- Situaciones Específicas
- Recursos contra la resolución

## Data Structure

```json
"article_number": {
            "title": "Artículo 1. Objeto y beneficiarios.",
            "title-trimmed": "Objeto y beneficiarios.",
            "content": "...",
            "pages": [
                2
            ],
            "chapter_number": "I",
            "chapter_name": "Objeto y ámbito de aplicación",
            "summary": "...",
            "topic": "Obligaciones, Control y Situaciones Especiales",
            "subtopic": "Recursos contra la resolución",
            "topic_score": 0.6534093022346497,
            "user_query_score": 0.6540930271148682
}
