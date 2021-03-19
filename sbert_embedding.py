from sentence_transformers import SentenceTransformer


class SentenceEmbedding:

    def __init__(self, model_path):
        self.model = SentenceTransformer(model_path)

    def encode(self, content):
        sentence_embeddings = self.model.encode(content)
        return sentence_embeddings
