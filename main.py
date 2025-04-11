"""
This is main entry point to our processing.
It applies all the processing steps in sequence.
It allows to reuse the cached summaries (they are the expensive pipeline part)
to change the user query and get new abstract summaries.

Processing Steps:
step1_summarize_articles -> we obtain the documents_summarized.json
step2_bert2topics -> Compute topic/subtopic scores and obtain documents_summarized_with_topics.json
step3_generate_final_summaries -> Generate the final summaries in markdown format and export to pdf
"""

import os
import json
import click
import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from step1_summarize_articles import generate_json_summary
from step2_bert2topics import compute_bert_scores
from step3_generate_final_summaries import process_documents


@click.command()
@click.option(
    "--topk",
    type=click.IntRange(min=1),
    default=None,
    help="Top-k summaries to include. Defaults to None which includes all",
)
@click.option(
    "--threshold",
    type=float,
    default=0.0,
    help="Minimum article score when processing user query.",
)
@click.option(
    "--user-query", type=str, default=None, help="Query string for topic modeling."
)
@click.option(
    "--regenerate-summaries", is_flag=True, help="Use cached summary JSON if available."
)
def main(topk, threshold, user_query, regenerate_summaries):
    # Our pdfs :)
    pdf_files = [
        "documents/ayudas_21-22.pdf",
        "documents/ayudas_22-23.pdf",
        "documents/ayudas_23-24.pdf",
        "documents/ayudas_24-25.pdf",
        "documents/ayudas_25-26.pdf"
    ]

    # **** STEP 1 Generating all summaries for our pdfs ***

    summaries_json = "documents_summarized.json"
    if regenerate_summaries:
        # Create model for summarization
        print("(1) Generating summaries from scratch")
        print("Loading Mistral-Nemo-Instruct-2407 model...")
        model_name = "mistralai/Mistral-Nemo-Instruct-2407"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model_name == "mistralai/Mistral-Nemo-Instruct-2407":
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
                )

            model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=quantization_config
        )
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16
            )

        main_dict = {}
        for pdf_path in pdf_files:
            print(f"Processing {pdf_path}...")
            base_name = os.path.basename(pdf_path)
            main_dict[base_name] = generate_json_summary(model, tokenizer, pdf_path)

        with open(summaries_json, "w", encoding="utf-8") as file:
            json.dump(main_dict, file, indent=2, ensure_ascii=False)

        print("All documents processed and saved to documents_summarized.json")
    else:
        print(f"(1) Using cached summaries from {summaries_json}")

    # *****************************

    # **** STEP 2 Using bert to add topics/subtopics score to articles ***
    print('(2) Computing BERT scores for topics/subtopics')
    bert_output_file = summaries_json.replace('.json', '_with_topics.json')
    data = compute_bert_scores(summaries_json, user_query)

    # Save the updated JSON to a file
    with open(bert_output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Updated JSON saved to {bert_output_file}")

    # *****************************

    # **** STEP 3 Generate final summaries ***
    print('(3) Generating final abstract summaries')
    process_documents(bert_output_file, topk=topk, use_user_query=user_query is not None, threshold=threshold)


if __name__ == '__main__':
    main()
