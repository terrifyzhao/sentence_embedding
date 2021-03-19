import torch
from transformers import BertTokenizerFast
from transformers import BertModel
import numpy as np
from tqdm import tqdm

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


class SentenceEmbedding:

    def __init__(self, model_path):

        self.model = BertModel.from_pretrained(model_path).to(device)
        self.tokenizer = BertTokenizerFast.from_pretrained(model_path)

    def encode(self, content, batch_size=256, max_length=None, padding='max_length'):

        pool_outs = None
        if isinstance(content, list) and len(content) > batch_size:
            for epoch in tqdm(range(len(content) // batch_size + 1)):
                batch_content = content[epoch * batch_size:(epoch + 1) * batch_size]
                if batch_content:
                    pool_out = self._embedding(batch_content, max_length, padding)
                    if pool_outs is None:
                        pool_outs = pool_out
                    else:
                        pool_outs = np.concatenate([pool_outs, pool_out], axis=0)
            return pool_outs
        else:
            return self._embedding(content, max_length, padding)

    def _embedding(self, content, max_length, padding):
        if max_length is None:
            if isinstance(content, str):
                max_length = len(content)
            else:
                length = [len(c) for c in content]
                max_length = max(length)

        inputs = self.tokenizer(content,
                                return_tensors="pt",
                                truncation=True,
                                padding=padding,
                                max_length=max_length)

        outputs = self.model(**inputs.to(device), output_hidden_states=True)
        pool_out = outputs[1].cpu().data.numpy()

        return pool_out
