import torch
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from topics import topics
from typing import Optional



DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = 'mps'


class BertScorer:
    def __init__(self, topics: list[tuple[str, str]]):  # TODO: Topics type?
        model_name = "mrm8488/modernbert-embed-base-ft-sts-spanish-matryoshka-768-64-5e"
        self.model = SentenceTransformer(model_name, device=DEVICE)

        # TODO: Decide how relates with subtopics
        embeddings = self.model.encode([f'{topic}: {subtopic}' for (topic, subtopic) in topics])
        self.topics_embeddings = {topic: emb for topic, emb in zip(topics, embeddings)}

    def get_topics_score(self, query: str):
        query_emb = self.model.encode(query)
        similarities = self._get_scores(query_emb, self.topics_embeddings)
        return similarities
    
    def pair_score(self, a, b):
        a = self.model.encode(a)
        b = self.model.encode(b)
        return self.model.similarity(a, b)[0]
    
    def _get_scores(self, query_emb: np.ndarray, topics_embs: dict):
        similarities = self.model.similarity(query_emb, np.vstack(list(topics_embs.values())))[0]
        return {topic: similarity.item() for topic, similarity in zip(topics_embs.keys(), similarities)}


def compute_bert_scores(input_file='./documents_summarized.json', output_file='./documents_summarized_with_topics.json', user_query=None):
    scorer = BertScorer(topics)
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for doc, articles in data.items():
        for article_id, article in articles.items():
            title_trimmed = f'{article["chapter_name"]}: {article["title-trimmed"]}'
            topics_scores = scorer.get_topics_score(title_trimmed)

            print()
            print(title_trimmed)
            print()

            user_query_score = None
            if user_query is not None:
                user_query_score = scorer.pair_score(title_trimmed, user_query).item()
                # print(title_trimmed, user_query, user_query_score)

            best_topic, best_score = max(topics_scores.items(), key=lambda item: item[1])

            # Add the best topic to the article
            article["topic"] = best_topic[0]
            article["subtopic"] = best_topic[1]
            article["topic_score"] = best_score
            article["user_query_score"] = user_query_score

    # Save the updated JSON to a file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Updated JSON saved to {output_file}")


if __name__ == '__main__':
    # Testing
    input_file = "./documents_summarized.json"  # Replace with your JSON file path
    output_file = "./documents_summarized_with_topics.json"  # Replace with your desired output file path
    user_query = 'Me gustaría obtener los requisitos académicos para obtener una beca.'

    compute_bert_scores(input_file, output_file, user_query)