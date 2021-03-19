import torch
from transformers import BertTokenizerFast
from transformers import BertModel

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


class SentenceEmbedding:

    def __init__(self, model_path):
        self.model = BertModel.from_pretrained(model_path).to(device)
        self.tokenizer = BertTokenizerFast.from_pretrained(model_path)

    def encode(self, content):
        inputs = self.tokenizer(content, return_tensors="pt")
        output = self.model(**inputs.to(device))[1].cpu().data.numpy().flatten()

        return output
