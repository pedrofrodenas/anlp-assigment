import torch
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from topics import topics



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
    
    def _get_scores(self, query_emb: np.ndarray, topics_embs: dict):
        similarities = self.model.similarity(query_emb, np.vstack(list(topics_embs.values())))[0]
        return {topic: similarity.item() for topic, similarity in zip(topics_embs.keys(), similarities)}


if __name__ == '__main__':
    # Query text for tests
    # query = 'Rendimiento académico en el curso anterior.'
    # scorer = BertScorer(topics)

    # # Compute embeddings:
    # topics_scores = scorer.get_topics_score(query)

    # for topic, score in topics_scores.items():
    #     print(f"Similarity with {topic}: {score:.4f}")

    # # Determine the best matching topic
    # best_topic = max(topics_scores, key=topics_scores.get)
    # print(f"\nThe query '{query}' is most similar to the topic: '{best_topic}'")

    # From JSON file
    scorer = BertScorer(topics)
    input_file = "./documents_summarized.json"  # Replace with your JSON file path
    output_file = "./documents_summarized_with_topics.json"  # Replace with your desired output file path

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for doc, articles in data.items():
        for article_id, article in articles.items():
            title_trimmed = article["title-trimmed"]
            topics_scores = scorer.get_topics_score(title_trimmed)
            best_topic, best_score = max(topics_scores.items(), key=lambda item: item[1])

            # Add the best topic to the article
            article["topic"] = best_topic[0]
            article["subtopic"] = best_topic[1]
            article["topic_score"] = best_score

    # Save the updated JSON to a file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Updated JSON saved to {output_file}")