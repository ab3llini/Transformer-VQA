from pytorch_transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenize = lambda text: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

s = 'vegetablesccoli'

tk = tokenize(s)
print(tk)
dec = tokenizer.convert_ids_to_tokens(tk)
print(dec)
