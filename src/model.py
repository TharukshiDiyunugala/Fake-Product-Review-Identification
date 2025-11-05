from transformers import BertTokenizer, BertForSequenceClassification

def load_model(model_name='bert-base-uncased', num_labels=2):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return tokenizer, model
