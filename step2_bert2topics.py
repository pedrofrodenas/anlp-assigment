import torch
import numpy as np
from sentence_transformers import SentenceTransformer



DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = 'mps'


class BertScorer:
    def __init__(self, topics, subtopics):  # TODO: Topics type?
        model_name = "mrm8488/modernbert-embed-base-ft-sts-spanish-matryoshka-768-64-5e"
        self.model = SentenceTransformer(model_name, device=DEVICE)

        # TODO: Decide how relates with subtopics
        embeddings = self.model.encode([f'{topic} {desc}' for topic, desc in topics.items()])
        self.topics_embeddings = {topic: emb for topic, emb in zip(topics.keys(), embeddings)}

        self.subtopics_embeddings = {}
        for topic in topics.keys():
            embeddings = self.model.encode([f'{topic} {desc}' for topic, desc in subtopics[topic].items()])
            self.subtopics_embeddings[topic] = {topic: emb for topic, emb in zip(subtopics[topic].keys(), embeddings)}

    def get_topics_score(self, query: str):
        query_emb = self.model.encode(query)
        similarities = self._get_scores(query_emb, self.topics_embeddings)
        return similarities
    
    def _get_scores(self, query_emb: np.ndarray, topics_embs: dict):
        similarities = self.model.similarity(query_emb, np.vstack(list(topics_embs.values())))[0]
        return {topic: similarity.item() for topic, similarity in zip(topics_embs.keys(), similarities)}
    
    def get_subtopic_score(self, topic: str, query: str):
        assert topic in self.subtopics_embeddings, f'No subtopics defined for topic {topic}'
        query_emb = self.model.encode(query)
        similarities = self._get_scores(query_emb, self.subtopics_embeddings[topic])
        return similarities


if __name__ == '__main__':
    topics = {
        "Información General y Tipos de Beca": "",
        "Estudios Cubiertos por la Beca": "",
        "Requisitos para Solicitar la Beca": "",
        "Proceso de Solicitud y Tramitación": "",
        "Obligaciones, Control y Situaciones Especiales": "",
    }

    subtopics = {
        "Información General y Tipos de Beca": {
            "Objeto de la Convocatoria y Financiación": "",
            "Tipos de Cuantías": "",
            "Beca de Matrícula": "",
            "Beca Básica": "",
            "Cuantías Adicionales": "",
        },

        "Estudios Cubiertos por la Beca": {
            "Enseñanzas No Universitarias": "",
            "Enseñanzas Universitarias": "",
        },

        "Requisitos para Solicitar la Beca": {
            "Requisitos Generales": "",
            "Requisitos Económicos": "",
            "Requisitos Académicos": "",
        },

        "Proceso de Solicitud y Tramitación": {
            "Presentación de Solicitud": "",
            "Documentación y Autorizaciones": "",
            "Revisión, Subsanación y Alegaciones": "",
            "Órganos de Selección y Tramitación": "",
            "Resolución, Notificación y Consulta de Estado": "",
            "Pago de la Beca": "",
        },

        "Obligaciones, Control y Situaciones Especiales": {
            "Obligaciones de los Becarios": "",
            "Control, Verificación y Reintegro": "",
            "Compatibilidades e Incompatibilidades con otras ayudas": "",
            "Situaciones Específicas": "",
            "Recursos contra la resolución": "",
        }
    }

    # Query text
    query_chapter = "CAPÍTULO V: Requisitos de carácter académico"
    query_article = 'Artículo 24. Rendimiento académico en el curso anterior.'

    scorer = BertScorer(topics, subtopics)

    # Compute embeddings:
    topics_scores = scorer.get_topics_score(query_chapter)

    for topic, score in topics_scores.items():
        print(f"Similarity with {topic}: {score:.4f}")

    # Determine the best matching topic
    best_topic = max(topics_scores, key=topics_scores.get)
    print(f"\nThe query '{query_chapter}' is most similar to the topic: '{best_topic}'")

    subtopics_scores = scorer.get_subtopic_score(best_topic, query_article)

    for subtopic, score in subtopics_scores.items():
        print(f"Similarity with {subtopic}: {score:.4f}")

    # Determine the best matching subtopic
    best_subtopic = max(subtopics_scores, key=subtopics_scores.get)
    print(f"\nThe query '{query_article}' is most similar to the subtopic: '{best_subtopic}' of topic '{best_topic}'")