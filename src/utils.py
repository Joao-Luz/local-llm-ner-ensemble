import numpy as np

from datasets import Dataset
from nltk.corpus.reader.conll import ConllCorpusReader
from pathlib import Path
from itertools import combinations, product
from collections import defaultdict
import json

import networkx as nx

def tokenize_and_align_labels(batch, tokenizer, label2idx=None):
    if not isinstance(batch['tokens'][0], list):
        batch['tokens'] = [batch['tokens']]
        batch['ner_tags'] = [batch['ner_tags']]
    
    tokenized_inputs = tokenizer(batch['tokens'], truncation=True, is_split_into_words=True, max_length=512)

    labels = []
    for i, label in enumerate(batch['ner_tags']):
        if isinstance(label[0], str):
            label = [label2idx[l] for l in label]

        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx]+1 if label[word_idx] % 2 else label[word_idx])
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs['labels'] = labels
    return tokenized_inputs

def read_dataset(file_name):
    root = str(Path(file_name).parent)
    fileid = Path(file_name).name
    reader = ConllCorpusReader(root=root, columntypes=['words', 'pos'], fileids=[fileid])
    tagged_sentences = reader.tagged_sents()

    dataset = Dataset.from_dict({
        'tokens': [[token for token,_ in sentence] for sentence in tagged_sentences],
        'ner_tags': [[label for _,label in sentence] for sentence in tagged_sentences]
    }) 

    return dataset

def write_dataset(dataset, file, id2label=None):
    Path(file).parent.mkdir(parents=True, exist_ok=True)

    with open(file, 'w') as f:
        for sentence,labels in zip(dataset['tokens'], dataset['ner_tags']):
            for token,label in zip(sentence, labels):
                if id2label is None:
                    f.write(token + '\t' + label + '\n')
                else:
                    f.write(token + '\t' + id2label[label] + '\n')
            f.write('\n')

def iterate_entities(tokens, labels):
    current_entity = None
    current_entity_type = None
    start_index = None

    for i, (token, label) in enumerate(zip(tokens, labels)):
        if label == 'O':
            tag = label
        else:
            tag, entity_type = label.split('-')
        
        if tag == 'B':  # Beginning of a new entity
            if current_entity:
                yield current_entity_type, current_entity, start_index, i
            current_entity = token
            current_entity_type = entity_type
            start_index = i
        elif tag == 'I' and current_entity:  # Inside an entity
            if token.startswith('##'):
                current_entity += token[2:]
            else:
                current_entity += ' ' + token
        else:  # Outside an entity
            if current_entity:
                yield current_entity_type, current_entity, start_index, i
                current_entity = None
                current_entity_type = None
                start_index = None

    # If an entity spans until the end of the sentence
    if current_entity:
        yield current_entity_type, current_entity, start_index, len(tokens)

def convert_predictions(predicted, true, idx2label=None):
    predicted = np.argmax(predicted, axis=2)

    def to_label(i):
        if idx2label is None: return str(i)
        else: return idx2label[i]

    predicted = [[to_label(p) for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predicted, true)]

    true = [[to_label(l) for l in label if l != -100] for label in true]
    return predicted, true

def convert_bert_tokens_to_regular(tokens, labels, word_ids, original_tokens=None, idx2label=None):
    regular_tokens = []
    regular_labels = []

    prev = -1
    for word_id,token,label in zip(word_ids, tokens, labels):
        if prev == word_id:
            if original_tokens is None:
                regular_tokens[-1] += token[2:] if token.startswith('##') else token

        else:
            if original_tokens is not None:
                token = original_tokens[word_id]
            regular_tokens.append(token)

            if idx2label is not None:
                if label < 0:
                    label = 0

                regular_labels.append(idx2label[label])
            else:
                regular_labels.append(label)

            # Tokenizer may skip a token
            if prev != word_id-1:
                regular_tokens.append(original_tokens[word_id-1])
                regular_labels.append('O')

            prev = word_id

    return regular_tokens, regular_labels

def group_sequences(sequences):
    G = nx.Graph()
    
    for i, seq in enumerate(sequences):
        G.add_node(i, sequence=seq)
        
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            if set(sequences[i]).intersection(sequences[j]):
                G.add_edge(i, j)

    grouped_sequences = []
    for component in nx.connected_components(G):
        group = [sequences[i] for i in component]
        grouped_sequences.append(group)
    
    return grouped_sequences

def subsets_are_disjoint(subsets):
    seen_elements = set()
    for subset in subsets:
        if seen_elements.intersection(subset):
            return False
        seen_elements.update(subset)
    return True

def non_intersecting_combinations(subsets):
    all_combinations = []
    for r in range(1, len(subsets) + 1):
        for combo in combinations(subsets, r):
            if subsets_are_disjoint(combo):
                all_combinations.append(combo)
    return all_combinations

def mention_index(sentence, mention, start=0):
    tokens = sentence.split()
    entity_tokens = mention.split()
    ent_len = len(entity_tokens)
    for i in range(start, len(tokens) - len(entity_tokens) + 1):
        if tokens[i:i+ent_len] == entity_tokens:
            return i

def group_entities(entities):
    # String match entities and save spans
    span_map = defaultdict(list)
    for i,e in enumerate(entities):
        start = e['start']
        end = e['end']
        span_map[(tuple(range(start, end)))].append(i)

    # Group entities that share spans
    groups = group_sequences(sorted(list(span_map.keys()), key=len, reverse=True))

    # Get every combination of non intersecting entities and types
    typed_groups = []
    for group in groups:
        combos = non_intersecting_combinations(group)
        combos = [[span_map[span] for span in sorted(spans, key=lambda x: x[0])] for spans in combos]
        combos = sorted(combos, key=lambda x: sum([len(m) for m in x]))

        typed_combos = []
        for combo in combos:
            all_combinations = list(product(*combo))
            typed_combos.extend(all_combinations)
        typed_groups.append(typed_combos)

    return typed_groups

def bio_to_sentence_spans(tokens, labels):
    entities = []
    for type,mention,start,end in iterate_entities(tokens, labels):
        s = sum(len(t) for t in tokens[:start]) + start
        e = s + len(mention)

        entities.append({
            'mention': mention,
            'type': type,
            'start': s,
            'end': e
        })

    return entities

def sentence_spans_to_bio(sentence, entities):
    tokens = sentence.split()
    labels = ['O' for _ in tokens]

    offset = 0
    prev_entity = None
    for i,token in enumerate(tokens):
        for entity in entities:
            if offset >= entity['start'] and offset < entity['end']:
                if entity == prev_entity:
                    labels[i] = 'I-' + entity['type']
                else:
                    labels[i] = 'B-' + entity['type']
                
                prev_entity = entity
        
        offset += len(token) +1

    return labels

def save_to_json(data, file_dir):
    Path(file_dir).parent.mkdir(parents=True, exist_ok=True)
    with open(file_dir, 'w') as f:
        json.dump(data, f, indent='\t', ensure_ascii=False)

def load_json(file_dir):
    if Path(file_dir).is_file():
        with open(file_dir) as f:
            return json.load(f)

def conll_labels(tokens, entities):
    tokens = [t.lower() for t in tokens]
    labels = ['O'] * len(tokens)
    for entity in entities:
        mention = entity['mention']
        if not mention.rstrip():
            continue
        type = entity['type']
        entity_tokens = mention.lower().split()
        
        for i in range(len(tokens) - len(entity_tokens) + 1):
            ent_len = len(entity_tokens)
            if tokens[i:i+ent_len] == entity_tokens and all(l == 'O' for l in labels[i:i+ent_len]):
                labels[i] = f'B-{type}'
                for j in range(1, ent_len):
                    labels[i+j] = f'I-{type}'
    
    return labels

def divide_in_batches(items, batch_size):
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]