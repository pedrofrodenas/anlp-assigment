from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn.functional import cosine_similarity


DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = 'mps'


class BertScorer:
    def __init__(self, topics, subtopics):  # TODO: Topics type?
        model_name = 'dccuchile/bert-base-spanish-wwm-cased'  # BETO cased model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, device_map=DEVICE)

        # TODO: Decide how relates with subtopics
        self.topics_embeddings = {topic: self.get_embedding(f'{topic}: {desc}')
                                  for topic, desc in topics.items()}

        self.subtopics_embeddings = {}
        for topic in topics.keys():
            print(subtopics[topic])
            self.subtopics_embeddings[topic] = {topic: self.get_embedding(f'{topic}: {desc}')
                                                for topic, desc in subtopics[topic].items()}

    # NOTE: Currently NOT in batches
    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        assert inputs['input_ids'].shape[0] == 1, 'Batch inference not supported right now'

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # TODO MAYBE NORMALIZE BERT EMBEDDINGS?

        # Mean pooling to similarity
        # To do so evaluate padding length and mask it
        embeddings = outputs.last_hidden_state
        mask = inputs['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float()
        # print(mask)
        masked_embeddings = embeddings * mask
        summed = torch.sum(masked_embeddings, 1)
        summed_mask = torch.clamp(mask.sum(1), min=1e-9)
        emb = summed / summed_mask
        return emb

    # Currently using mean pooling for scoring
    # TODO: MAKE IT BATCHED
    def get_topics_score(self, query):  # TODO: Max length? check beto context window
        query_emb = self.get_embedding(query)
        similarities = self._get_scores(query_emb, self.topics_embeddings)
        return similarities
    
    def _get_scores(self, query_emb, topics_embs):
        return {topic: cosine_similarity(query_emb, emb).item() for topic, emb in topics_embs.items()}
    
    def get_subtopic_score(self, topic, query):
        assert topic in self.subtopics_embeddings, f'No subtopics defined for topic {topic}'
        query_emb = self.get_embedding(query)
        similarities = self._get_scores(query_emb, self.subtopics_embeddings[topic])
        return similarities


if __name__ == '__main__':
    topics = {
        "REQUISITOS": "Criterios de admisión y elegibilidad",
        "BECAS": "Información sobre becas y ayudas financieras"
    }

    subtopics = {
        'REQUISITOS': {
            'Requisitos economicos': "",  # TODO
            'Requisitos academicos': "",
        },

        'BECAS': {
            'Verificación de las becas': "",
        }
    }

    # Query text
    query_chapter = "CAPÍTULO V: Requisitos de carácter académico"
    query_article = 'Artículo 24. Rendimiento académico en el curso anterior.'

    scorer = BertScorer(topics, subtopics)

    # Compute embeddings
    topics_scores = scorer.get_topics_score(query_chapter)

    # Output similarity scores
    for topic, score in topics_scores.items():
        print(f"Similarity with {topic}: {score:.4f}")

    # Determine the best matching topic
    best_topic = max(topics_scores, key=topics_scores.get)
    print(f"\nThe query '{query_chapter}' is most similar to the topic: '{best_topic}'")

    subtopics_scores = scorer.get_subtopic_score(topic, query_article)

    # Determine the best matching subtopic
    best_subtopic = max(subtopics_scores, key=subtopics_scores.get)
    print(f"\nThe query '{query_article}' is most similar to the subtopic: '{best_subtopic}' of topic '{best_topic}'")