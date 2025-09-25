import torch
from utils import iterate_entities, divide_in_batches
import json
import regex as re

ZERO_SHOT_PROMPT = """ \
Você é um linguista especialista em reconhecimento de entidades nomeadas e 
deve identificar entidades nomeadas em textos em português.

Veja os seguintes tipos de entidades nomeadas:

{label_descriptions}

Encontre entidades nomeadas dos tipos especificados e responda como uma lista
 de objetos json como no seguinte formato:
 
 [
    {{
        "mention": "menção_1",
        "type": "tipo_1"
    }},
    {{
        "mention: "menção_2",
        "type": "tipo_2"
    }}
]

Caso não existam entidades, responder como uma lista json vazia.

Agora forneça as entidades da seguinte sentença no formato json especificado. 
(forneça APENAS AS ENTIDADES E SEUS TIPOS e mais nenhuma informação)

Sentença: {sentence}
Entidades:
"""

FEW_SHOT_PROMPT = """ \
Você é um linguista especialista em reconhecimento de entidades nomeadas e 
deve identificar entidades nomeadas em textos em português.

Veja os seguintes tipos de entidades nomeadas:

{label_descriptions}

Encontre entidades nomeadas dos tipos especificados e responda como uma lista
de objetos JSON como no seguinte formato:
 
 [
    {{
        "mention": "menção_1",
        "type": "tipo_1"
    }},
    {{
        "mention: "menção_2",
        "type": "tipo_2"
    }}
]

Caso não existam entidades, responder como uma lista JSON vazia.

Agora veja exemplos de reconhecimento de entidades no formato JSON:

{examples}

Agora forneça as entidades da seguinte sentença no formato JSON especificado. 
(forneça APENAS AS ENTIDADES E SEUS TIPOS e mais nenhuma informação)

Sentença: {sentence}
Entidades:
"""

def parse_extraction_response(response, sentence, existing_labels, max_entity_length=100):
    entities = []
    pattern = r'"mention": "([^"]+)"[^"]*"type": "([^"]+)"'
    for m in re.finditer(pattern, response):
        mention = m.group(1).strip()
        type = m.group(2).strip()
        
        if type in existing_labels:
            for m in re.finditer(re.escape(mention.lower()), sentence.lower()):
                start, end = m.span()
                entities.append({
                    'mention': mention, 
                    'type': type, 
                    'start': start, 
                    'end': end
                })

    return entities

def format_examples(examples):
    def format_entities(si):
        ents = [{'mention': m, 'type': t} 
                for t,m,_,_ in iterate_entities(si['tokens'], si['ner_tags'])]
        return json.dumps(ents, ensure_ascii=False, indent=True)
    
    examples = [f'Sentence: {" ".join(si["tokens"])}\nEntities:\n{format_entities(si)}' 
                for si in examples]
    
    return '\n\n'.join(examples)

def format_descriptions(label_descriptions):
    descriptions = [f'{type}: {description}' for type,description in label_descriptions.items()]
    return '\n'.join(descriptions)

def extract_entities(sentences, model, label_descriptions, 
                     labeled=[], max_few_shot=5, temperature=0.8):
    prompts = []
    for sentence in sentences:
        template = ZERO_SHOT_PROMPT if max_few_shot == 0 else FEW_SHOT_PROMPT

        # Only random for now
        top_k_indices = torch.randperm(len(labeled))[:max_few_shot]
        examples = labeled.select(top_k_indices)
        examples_str = format_examples(examples)

        descriptions_str = format_descriptions(label_descriptions)

        prompt = template.format(sentence=sentence, label_descriptions=descriptions_str,
                                 examples=examples_str)
        prompts.append(prompt)
    
    outputs = model.query(prompts, temperature) if len(prompts) > 0 else []

    entities = []
    for sentence,response in zip(sentences, outputs):
        sentence_entities = parse_extraction_response(response, sentence, label_descriptions)
        entities.append(sentence_entities)

    return entities