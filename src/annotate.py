import json
import pickle

from extract import extract_entities
from vote import vote_for_entities
from disambiguate import disambiguate

from utils import (
    group_entities,
    load_json,
    read_dataset,
    save_to_json,
    sentence_spans_to_bio,
    write_dataset
)
from argparse import ArgumentParser
from transformers import set_seed
from tqdm import tqdm
from pathlib import Path

from llm import LLM

def parse_arguments():
    parser = ArgumentParser("Annotate a set of sentences via an LLM ensemble.")

    parser.add_argument('--llms', type=str, nargs='+', default=['llama3.1', 'qwen2', 'gemma2', 'phi3:14b', 'mistral'])
    parser.add_argument('--extraction_llms', required=False, type=str, nargs='+', default=None)
    parser.add_argument('--voting_llms', required=False, type=str, nargs='+', default=None)
    parser.add_argument('--best_fit_llms', required=False, type=str, nargs='+', default=None)

    parser.add_argument('--unlabeled_data', type=str)
    parser.add_argument('--labeled_data', type=str)

    parser.add_argument('--dataset_info', type=str)

    parser.add_argument('--group_cache_file', type=str, default=None)
    parser.add_argument('--outdir', type=str)
    parser.add_argument('--conll_outfile', type=str, default='annotations.conll')
    parser.add_argument('--voted_outfile', type=str, default='entities')
    parser.add_argument('--grouped_outfile', type=str, default='grouped')
    parser.add_argument('--selected_outfile', type=str, default='selected')

    parser.add_argument('--batch_size', type=int, default=50)

    parser.add_argument('--max_best_fit_options', type=int, default=5)
    parser.add_argument('--max_few_shot', type=int, default=5)
    parser.add_argument('--max_entity_length', type=int, default=100)
    parser.add_argument('--temperature', default='0.0')
    parser.add_argument('--weights', type=str)

    parser.add_argument('--seed', type=str, default=42)

    return parser.parse_args()

def merge_llms_extractions(extracted, voted, temperatures):
    for i,sentence_info in enumerate(voted):
        for llm in extracted:
            for entity in extracted[llm][i]['entities']:
                start = entity['start']
                end = entity['end']

                indices = [j for j,e in enumerate(sentence_info['entities'])
                           if e['start'] == start and e['end'] == end and e['type'] == entity['type']]
                if indices:
                    ent = sentence_info['entities'][indices[0]]
                    if 'votes' not in ent:
                        ent['votes'] = {}

                    if 'source' not in ent:
                        ent['source'] = {}

                    if llm not in ent['source']:
                        ent['source'][llm] = [temperatures[llm]]

                    if temperatures[llm] not in ent['source'][llm]:
                        ent['source'][llm].append(temperatures[llm])
                else:
                    sentence_info['entities'].append(entity)
                    sentence_info['entities'][-1]['votes'] = {}
                    sentence_info['entities'][-1]['source'] = { llm: [temperatures[llm]] }
    return voted

def group_entities_for_all_sentences(sentences, extraction_llms, voting_llms, temperatures):
    grouped = sentences.copy()

    for sentence_info in tqdm(grouped, desc='Grouping entities'):
        if 'groups' in sentence_info:
            continue

        entities = sentence_info['entities']

        def filter_entity(entity):
            was_extracted = any(temperatures[llm] in entity['source'][llm] 
                                for llm in extraction_llms if llm in entity['source'])
            # valid = [v for m,v in entity['votes'].items() if m in voting_llms and v]
            valid = [v for m,v in entity['votes'].items() if m in voting_llms]
            was_voted = len(valid) > len(voting_llms)//2

            return was_extracted and was_voted

        # Only select entities that were extracted and voted by valid llms
        entities = list(filter(filter_entity, entities))
        groups = group_entities(entities)

        for i,group in enumerate(groups):
            # Sort by the length of the mentions in the group
            group = sorted(
                group,
                key=lambda x: sum(len(entities[e]['mention']) for e in x), 
                reverse=True
            )[:args.max_best_fit_options]
            group = [[{k: v for k,v in entities[e].items() if k in ['mention', 'start', 'end', 'type']} 
                      for e in option] for option in group]
            groups[i] = { 'options': { str(j): options for j,options in enumerate(group) }, 'votes': {} }

        sentence_info['groups'] = groups
        sentence_info['entities'] = entities

    return grouped

def select_best_fits(sentences, disambiguation_llms):
    best_fits = sentences.copy()
    for sentence_info in best_fits:
        selected_entities = []
        groups = sentence_info['groups']

        for entity_groups in groups:
            # Chose option with most votes
            votes = [vote for llm,vote in entity_groups['votes'].items() if llm in disambiguation_llms]
            best_fit = max(set(votes), key=votes.count)
            if best_fit == -1:
                continue
            else:
                selected_entities.extend([e for e in entity_groups['options'][str(best_fit)]])
        
        sentence_info['entities'] = selected_entities
        del sentence_info['groups']

    return best_fits

def convert_to_conll_dataset(sentences):
    dataset = {'tokens': [], 'ner_tags': []}
    for sentence_info in sentences:
        sentence = sentence_info['sentence']
        entities = sentence_info['entities']

        tokens = sentence.split()
        ner_tags = sentence_spans_to_bio(sentence, entities)

        dataset['tokens'].append(tokens)
        dataset['ner_tags'].append(ner_tags)

    return dataset

def to_option_tuple(options):
    options_tuple = []
    for option in options:
        option_tuple = []
        for entity in option:
            option_tuple.append((entity['start'], entity['end'], entity['type']))

        option_tuple = tuple(option_tuple)
        options_tuple.append(option_tuple)

    options_tuple = tuple(options_tuple)
    return options_tuple

def main(args):
    set_seed(args.seed)
    extraction_llms = args.extraction_llms or args.llms
    voting_llms = args.voting_llms or args.llms
    best_fit_llms = args.best_fit_llms or voting_llms

    print(f'Train data is "{args.labeled_data}" and unlabeled is "{args.unlabeled_data}"')
    print(f'Extraction: {extraction_llms}')
    print(f'Voting: {voting_llms}')
    print(f'Disambiguation: {best_fit_llms}')

    try:
        temp = float(args.temperature)
        temperatures = {llm: temp for llm in extraction_llms}
    except:
        with open(args.temperature) as f:
            temperatures = json.load(f)

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    unlabeled = read_dataset(args.unlabeled_data)
    labeled = read_dataset(args.labeled_data)

    with open(args.dataset_info, 'r') as f:
        dataset_info = json.load(f)
        label_descriptions = dataset_info['label_descriptions']

    # Load extracted entities if they exist
    extracted = {}
    for llm in extraction_llms:
        temp = temperatures[llm] or 0.0
        json_file = f'{args.outdir}/temp={temp:.1f}/{llm.split('/')[-1]}.json'
        if Path(json_file).is_file():
            print(f'Loading entities from {json_file}')
            extracted[llm] = load_json(json_file)
        else:
            extracted[llm] = []

    # Extraction via LLMs
    for llm in extraction_llms:
        temp = temperatures[llm] if llm in temperatures else 0.0
        model = LLM(llm)

        processed_indices = [si['index'] for si in extracted[llm]]

        print(f'Skipped {len(processed_indices)}/{len(unlabeled)} sentences with LLM {llm}')

        indices_to_extract = [i for i in range(len(unlabeled)) if i not in processed_indices]
        data_to_extract = unlabeled.select(indices_to_extract)

        sentences = [' '.join(tokens) for tokens in data_to_extract['tokens']]
        entities = extract_entities(sentences, model, label_descriptions, 
                                    labeled, args.max_few_shot, temp)
        
        for i,sentence,sentence_entities in zip(indices_to_extract, sentences, entities):
            sentence_info = {
                'index': i,
                'sentence': sentence,
                'entities': sentence_entities
            }

            extracted[llm].append(sentence_info)

        extracted[llm].sort(key=lambda si: si['index'])

        json_file = f'{args.outdir}/temp={temp:.1f}/{llm.split('/')[-1]}.json'
        save_to_json(extracted[llm], json_file)

        model._delete_model()

    # Merge all entities to a single dict
    voted_file = f'{args.outdir}/{args.voted_outfile}.json'
    if Path(voted_file).is_file():
        print(f'Loading voted entities from {voted_file}')
        voted = load_json(voted_file)
    else:
        voted = [{'index': i, 'sentence': ' '.join(tokens), 'entities': []} 
                  for i,tokens in enumerate(unlabeled['tokens'])]

    voted = merge_llms_extractions(extracted, voted, temperatures)
    
    # Vote if entities are valid
    for llm in voting_llms:
        print(f'Voting valid entities with {llm}')

        temp = temperatures[llm] or 0.0
        model = LLM(llm)

        # Filter out unnecessary entities to vote
        indices_to_vote = []
        data_to_vote = []
        for i,sentence_info in enumerate(voted):
            sentence_indices_to_vote = []
            sentence_data_to_vote = []
            for j,entity in enumerate(sentence_info['entities']):
                if llm in entity['votes']:
                    continue
                sentence_indices_to_vote.append(j)
                sentence_data_to_vote.append(entity)
            
            if sentence_indices_to_vote or sentence_data_to_vote:
                indices_to_vote.append((i, sentence_indices_to_vote))
                data_to_vote.append({
                    'sentence': sentence_info['sentence'], 
                    'entities': sentence_data_to_vote
                })

        # Do LLM voting
        votes = vote_for_entities(data_to_vote, model, label_descriptions,
                                  labeled, args.max_few_shot)

        # Save votes to correct entities
        for (i,entity_indices),entity_votes in zip(indices_to_vote, votes):
            for entity_index, entity_vote in zip(entity_indices, entity_votes):
                voted[i]['entities'][entity_index]['votes'][llm] = entity_vote

        model._delete_model()
        
        # Save current votes to disk
        save_to_json(voted, voted_file)

    # Group intersecting entities
    grouped_file = f'{args.outdir}/{args.grouped_outfile}.json'
    if Path(grouped_file).is_file():
        print(f'Loading groups from {grouped_file}')
        grouped = load_json(grouped_file)
    else:
        grouped = voted
        Path(grouped_file).parent.mkdir(parents=True, exist_ok=True)

    grouped = group_entities_for_all_sentences(grouped, extraction_llms, voting_llms, temperatures)

    # Vote for best fit
    group_cache_file = f'{args.outdir}/{args.group_cache_file}'
    if Path(group_cache_file).is_file():
        with open(group_cache_file, 'rb') as f:
            group_cache = pickle.load(f)
    else:
        Path(group_cache_file).parent.mkdir(parents=True, exist_ok=True)
        group_cache = {}
    
    for llm in best_fit_llms:
        temp = temperatures[llm] or 0.0
        model = LLM(llm)

        # Filter out unnecessary votes
        indices_to_disambiguate = []
        data_to_disambiguate = []
        cached = 0
        for i,sentence_info in enumerate(grouped):
            sentence_indices_to_disambiguate = []
            sentence_data_to_disambiguate = []
            for j,group in enumerate(sentence_info['groups']):
                options_tuple = (i, to_option_tuple(group['options'].values()))
                # If no options, answer is N/A
                if len(group['options']) == 0:
                    group['votes'][llm] = -1
                # If one option, answer is the one available
                elif len(group['options']) == 1:
                    group['votes'][llm] = 0
                # If the options were processed before by the current LLM, use the cached response
                elif options_tuple in group_cache and llm in group_cache[options_tuple]:
                    votes = group_cache[options_tuple]
                    group['votes'][llm] = votes[llm]
                    cached += 1
                # Should be voted
                else:
                    sentence_indices_to_disambiguate.append(j)
                    sentence_data_to_disambiguate.append(group)
            
            if sentence_indices_to_disambiguate or sentence_data_to_disambiguate:
                indices_to_disambiguate.append((i, sentence_indices_to_disambiguate))
                data_to_disambiguate.append({
                    'sentence': sentence_info['sentence'],
                    'groups': sentence_data_to_disambiguate
                })
        
        print(cached)
        # Do LLM disambiguation
        disambiguate_votes = disambiguate(data_to_disambiguate, model, label_descriptions)

        # Save disambiguation votes to the correct entity groups
        for (i,group_indices),group_votes in zip(indices_to_disambiguate, disambiguate_votes):
            for group_index, group_vote in zip(group_indices, group_votes):
                grouped[i]['groups'][group_index]['votes'][llm] = group_vote

                # Save to cache
                index = grouped[i]['index']
                group = grouped[i]['groups'][group_index]
                options_tuple = (index, to_option_tuple(group['options'].values()))

                if options_tuple not in group_cache:
                    group_cache[options_tuple] = {}
                group_cache[options_tuple][llm] = group_vote

        model._delete_model()

        with open(group_cache_file, 'wb') as f:
            pickle.dump(group_cache, f)

    save_to_json(grouped, grouped_file)

    # Select the options with the most votes
    selected = select_best_fits(grouped, disambiguation_llms=best_fit_llms)
    selected_file = f'{args.outdir}/{args.selected_outfile}.json'
    save_to_json(selected, selected_file)

    # Write final selection to .conll format
    conll_dataset = convert_to_conll_dataset(selected)
    print(f'Writting annotated data to "{args.outdir}/{args.conll_outfile}"')
    write_dataset(conll_dataset, f'{args.outdir}/{args.conll_outfile}')

if __name__ == '__main__':
    args = parse_arguments()
    main(args)