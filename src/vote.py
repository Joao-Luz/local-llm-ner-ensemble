import torch
from utils import iterate_entities, save_to_json
import json

ZERO_SHOT_PROMPT = """\
Você é um linguista especialista na língua portuguesa.

Considere a descrição do seguinte tipo de entidade nomeada:

{description}

Veja a seguinte sentença:

{sentence}

Responda com SIM ou NÃO e, em seguida, justifique: "{mention}" é uma entidade do tipo "{type}"?
"""

FEW_SHOT_PROMPT = """\
Você é um linguista especialista na língua portuguesa.

Considere a descrição do seguinte tipo de entidade nomeada:

{description}


Agora veja exemplos dessas entidades

{examples}


Veja a seguinte sentença:

{sentence}

Responda com SIM ou NÃO e, em seguida, justifique: "{mention}" é uma entidade do tipo "{type}"?
"""

def parse_voting_response(response):
    return response.lower().lstrip().startswith('sim')

def format_examples(labeled, max_few_shot, type):
    top_k_indices = torch.randperm(len(labeled))[:max_few_shot]
    
    examples = [m for si in labeled.select(top_k_indices) 
                for t,m,_,_ in iterate_entities(si['tokens'], si['ner_tags']) if t == type]
    
    return '\n'.join(examples)

def format_description(type, description):
    return f'{type}: {description}'

def vote_for_entities(sentences, model, label_descriptions, labeled=[],
                      max_few_shot=5, temperature=0.0):
    inputs = []
    for i,sentence_info in enumerate(sentences):
        for j,entity in enumerate(sentence_info['entities']):
            type = entity['type']
            template = ZERO_SHOT_PROMPT if max_few_shot == 0 else FEW_SHOT_PROMPT

            description_str = format_description(type, label_descriptions[type])

            examples_str = format_examples(labeled, max_few_shot, type)
            prompt = template.format(sentence=sentence_info['sentence'],
                                     mention=entity['mention'], type=entity['type'],
                                     description=description_str,
                                     examples=examples_str)
            inputs.append((i,j,prompt))

    prompts = [p for _,_,p in inputs]
    outputs = model.query(prompts, temperature) if len(prompts) > 0 else []

    votes = [[None for _ in si['entities']] for si in sentences]
    for (i,j,_),response in zip(inputs, outputs):
        vote = parse_voting_response(response)
        votes[i][j] = vote

    return votes
