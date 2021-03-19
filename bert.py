from transformers.models.bert import BertModel
from transformers import BertTokenizerFast

path = '/Users/joezhao/Documents/pretrain model/chinese_bert_L-12_H-768_A-12'
tokenizer = BertTokenizerFast.from_pretrained(path)
model = BertModel.from_pretrained(path)

encoding = tokenizer('敖德萨所多', return_tensors='pt')

res = model(encoding['input_ids'], encoding['attention_mask'])[1]
print(res)
