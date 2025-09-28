import os
import json
import pandas as pd
from tqdm import tqdm
from openie import openie
from medcat.cat import CAT
from multiprocessing import Pool
from openie import StanfordOpenIE
from sklearn.feature_extraction.text import TfidfVectorizer
from tools import cosine_similarity, get_bert_embeddings, compute_embeddings_batch


base_path = './data'
path_input = os.path.join(base_path, 'sample_input.json')
path_output = os.path.join(base_path, 'output.pkl')
embeddings_path_pickle = os.path.join(base_path, 'mimic_embeddings.pickle')
medcat_path = '~/Resources/medcat_snomed_unzip'

CORE_NLP_PATH = "~/Resources/stanford-corenlp-4.5.10"
os.environ["CORENLP_HOME"] = CORE_NLP_PATH

num_cores= 24
cat = CAT.load_model_pack(medcat_path)

# Patching CORENLP del exception
def safe_del(self):
    os.environ.pop("CORENLP_HOME", None)
openie.StanfordOpenIE.__del__ = safe_del

def aggregate(path_pickle, json_input_path, force_aggregating=True):
    if force_aggregating:
        source ,target= [], []
        json_input = json.load(open(json_input_path, 'r'))

        for entry in json_input:
            source.append(entry["source"])
            target.append(entry["target"])

        df_map = {'source':source, 'target':target}
        q_df = pd.DataFrame.from_dict(df_map)
        q_df.to_pickle(path_pickle)
    else:
        q_df = pd.read_pickle(path_pickle)
        print('Dataframe already constructed! Use force_tagging to update.')
    return q_df

def medcat_ner(df_path, columns, force_tagging=False, num_types=5):
    df = pd.read_pickle(df_path)
    if force_tagging or 'medcat_ner_%s' % columns[0] not in list(df_path):
        for column in columns:
            new_column = 'medcat_ner_%s' % column
            print('Tagging %s for names entities...'%column)
            entries = df[column].tolist()
            entities = []
            tmp_entities = {}
            for n, text in tqdm(enumerate(entries), total=len(entries)):
                try:
                    patient_entities = cat.get_entities(text)['entities']
                    tmp_entities[n] = patient_entities
                except:
                    print('Error extracting Entity in MedCat')
                    tmp_entities[n] = []
            # TF/IDF most heavy weighted types
            tmp_types = [' '.join([tmp_entities[n][entity]['type_ids'][0] for entity in tmp_entities[n]])# Why only the first type
                         for n in tmp_entities]
            tmp_types = [x for x in tmp_types if len(x)>1]
            vectorizer = TfidfVectorizer(max_features=num_types)
            vectorizer.fit_transform(tmp_types)
            allowed_types = set(vectorizer.get_feature_names_out().flatten())

            print('Medcat annotating for types: ', allowed_types)

            for doc_n in tmp_entities:
                cat_term = set()
                for entity in tmp_entities[doc_n]:
                    this_entity = tmp_entities[doc_n][entity]
                    typ = this_entity['type_ids']
                    detected_name = this_entity['detected_name'].replace('~', ' ')
                    if any([t in allowed_types for t in typ]):
                        cat_term.add(detected_name)
                entities.append(cat_term)
            df[new_column] = entities
        df.to_pickle(df_path)
    else:
        df = pd.read_pickle(df_path)
        print('Already tagged with named entities! Use force_tagging to re-tag.')
    return df


def open_ie(df_path, columns, force_tagging=False, num_cores=12):
    properties = {
        'openie.affinity_probability_cap': 2 / 3,
    }
    df = pd.read_pickle(df_path)
    if force_tagging or 'open_ie_%s' % columns[0] not in list(df_path):
        for column in columns:
            new_column = 'open_ie_%s' % column
            ner_column = 'medcat_ner_%s' % column
            print('Tagging %s for OpenIE...' % column)
            entries = df[column].tolist()
            triples = []
            with StanfordOpenIE(properties=properties, threads=num_cores, corenlp_path=CORE_NLP_PATH) as client: # StanfordOpenIE is not fork-safe, should be done sequentially
                for n, text in tqdm(enumerate(entries)):
                    triple = client.annotate(text)
                    triple = [t for t in triple if t['subject'] in df[ner_column][n]]
                    triples.append(triple)
            df[new_column] = triples
        df.to_pickle(df_path)
    else:
        print('Already tagged with OpenIE! Use force_tagging to re-tag.')
    return


def weighted_f_score(truth_prediction_facts:tuple):
    """
    :param truth_facts: list of truth triples [{'subject': 'a', 'relation': 'R', 'object': 'b'},...]
    :param prediction_facts: list of truth triples [{'subject': 'a', 'relation': 'R', 'object': 'b'},...]
    :return: tp, fp, fn, p, r, f
    """
    truth_facts, prediction_facts = truth_prediction_facts[0], truth_prediction_facts[1]
    tp, fp, fn = 0, 0, 0
    triple_tp, triple_fp, triple_fn = list(), list(), list()
    sub_obj_truth = set((f['subject'], f['object']) for f in truth_facts)
    sub_obj_prediction = set((f['subject'], f['object']) for f in prediction_facts)
    for t_fact in truth_facts:
        if (t_fact['subject'], t_fact['object']) not in sub_obj_prediction:
            fn+=1
        else:
            for p_fact in prediction_facts:
                if t_fact['subject'] == p_fact['subject']:
                    if t_fact['object'] == p_fact['object']:
                        tp+= cosine_similarity(embeddings_dict[t_fact['relation']], embeddings_dict[p_fact['relation']])
                        triple_tp.append(p_fact)
    for p_fact in prediction_facts:
        if (p_fact['subject'], p_fact['object']) not in sub_obj_truth:
            fp+=1
            triple_fp.append(p_fact)

    p = tp/(tp+fp) if tp!= 0 else 0
    r = tp / (tp + fn) if tp != 0 else 0
    f = 2*p*r / (p + r) if p !=0 and r !=0 else 0
    return float(f)

def get_wighted_f_score(df_path, columns, force_tagging=False, num_cores=12):
    df = pd.read_pickle(df_path)
    new_column = 'weighed_f1'
    source_column = f"open_ie_{columns[0]}"
    target_column = f"open_ie_{columns[1]}"

    if force_tagging or new_column not in list(df_path):
        with Pool(processes=num_cores) as pool:
            f1_scores = pool.map(weighted_f_score, [(getattr(row, source_column), getattr(row, target_column)) for row in df.itertuples(index=False)])
            df[new_column] = f1_scores
            print(f1_scores)
            df.to_pickle(df_path)
    else:
        print('Weighted F1 scores are already estimated! Use force_tagging to re-estimate.')



if __name__ =="__main__":
    re_estimate = True
    aggregate(path_output, path_input, force_aggregating=re_estimate)
    medcat_ner(path_output, ['source', 'target'], force_tagging=re_estimate, num_types=10)
    open_ie(path_output, ['source', 'target'], force_tagging=re_estimate)
    embeddings_dict = get_bert_embeddings(path_output, embeddings_path_pickle, ['source','target'], force_tagging=re_estimate)
    get_wighted_f_score(path_output, ['source', 'target'], force_tagging=re_estimate)
