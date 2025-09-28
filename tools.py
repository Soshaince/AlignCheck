import torch
import pickle
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel


def cosine_similarity(vec1, vec2):
    vec1 = np.asarray(vec1, dtype=np.float64)  # explicitly convert
    vec2 = np.asarray(vec2, dtype=np.float64)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def get_bert_embeddings(df_path, embeddings_path_pickle, columns, force_tagging=False):
    df = pd.read_pickle(df_path)
    relations = set()
    if force_tagging:
        for column in columns:
            triple_column = 'open_ie_%s' % column
            for triples in df[triple_column].tolist():
                for triple in triples: relations.add(triple['relation'])
        embeddings_dict = compute_embeddings_batch(list(relations), batch_size=32)
        with open(embeddings_path_pickle, 'wb') as wr:
            pickle.dump(embeddings_dict, wr)
    else:
        embeddings_dict = pickle.load(open(embeddings_path_pickle, 'rb'))
    return embeddings_dict


def compute_embeddings_batch(words, batch_size=32):
    embeddings_dict = {}
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device: ', device)
    model.to(device)
    model.eval()

    for i in range(0, len(words), batch_size):
        batch_words = words[i:i + batch_size]
        inputs = tokenizer(batch_words, padding=True, truncation=True, return_tensors="pt")
        inputs = {key: val.to(model.device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # shape: [batch_size, hidden_size]
        cls_embeddings = cls_embeddings.cpu()
        for word, embedding in zip(batch_words, cls_embeddings):
            embeddings_dict[word] = embedding
    return embeddings_dict