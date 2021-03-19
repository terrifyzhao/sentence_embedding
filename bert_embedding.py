import torch
from transformers import BertTokenizerFast
from transformers import BertModel
import numpy as np
from tqdm import tqdm

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


class SentenceEmbedding:

    def __init__(self, model_path, mode='avg'):

        self.model = BertModel.from_pretrained(model_path).to(device)
        self.tokenizer = BertTokenizerFast.from_pretrained(model_path)
        self.mode = mode

    def encode(self, content, batch_size=256, max_length=None, padding='max_length'):

        outputs = None
        if isinstance(content, list) and len(content) > batch_size:
            for epoch in tqdm(range(len(content) // batch_size + 1)):
                batch_content = content[epoch * batch_size:(epoch + 1) * batch_size]
                if batch_content:
                    output = self._embedding(batch_content, max_length, padding)
                    if outputs is None:
                        outputs = output
                    else:
                        outputs = np.concatenate([outputs, output], axis=0)
            return outputs
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

        if self.mode == 'cls':
            output = self.model(**inputs.to(device))[1].cpu().data.numpy()
        elif self.mode == 'avg':
            output = np.mean(self.model(**inputs.to(device))[0].cpu().data.numpy(), axis=1)
        elif self.mode == 'two_avg':
            output = self.model(**inputs.to(device), output_hidden_states=True)[2][-2:]
            output = np.mean(torch.cat(output, dim=1).cpu().data.numpy(), axis=1)
        return output
