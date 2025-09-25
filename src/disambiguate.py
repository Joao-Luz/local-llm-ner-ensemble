import regex as re

PROMPT = """ \
Você é um linguista especialista na língua portuguesa.",

Considere a descrição do(s) seguinte(s) tipo(s) de entidade nomeada:

{descriptions}

Veja a seguinte sentença:

{sentence}

Qual das seguintes combinações de menções é mais adequada na sentença:

{options}

Obs.: Responda APENAS com o número indicando a opção correta, seguida de uma
breve explicação, no seguinte formato JSON:

{{
    "option": "escolha",
    "explanation": "explicação para a escolha"
}}

Se nenhuma combinação estiver correta, selecione a opção indicada por "N/A":
"""

def parse_disambiguate_response(response, options):
    pattern = r'"option": \"?([0-9]+)'
    matches = re.findall(pattern, response)

    if len(matches) == 0:
        return -1
    else:
        choice = int(matches[0])-1
        return choice if choice < len(options) else -1

def format_descriptions(label_descriptions):
    descriptions = [f'{type}: {description}' for type,description in label_descriptions.items()]
    return '\n'.join(descriptions)

def format_options(options):
    options_list = []
    for entities in options:
        options_list.append(', '.join(f'"{e["mention"]}" ({e["type"]})' for e in entities))

    options_str = '\t' + '\n\t'.join(f'{i+1}) {opt}' for i,opt in enumerate(options))
    options_str += f'\n\t{len(options)+1}) N/A\n'

    return options_str

def disambiguate(sentences, model, label_descriptions, temperature=0.0):
    disambiguated = sentences.copy()
    inputs = []
    for i,sentence_info in enumerate(disambiguated):
        sentence = sentence_info['sentence']
        groups = sentence_info['groups']

        # Go through groups
        for j,group in enumerate(groups):
            options = list(group['options'].values())
            options_str = format_options(options)

            present_types = set([entity['type'] for entity_list in options for entity in entity_list])
            descriptions = {k: v for k,v in label_descriptions.items() if k in present_types}
            descriptions_str = format_descriptions(descriptions)

            prompt = PROMPT.format(descriptions=descriptions_str, sentence=sentence, options=options_str)

            inputs.append((i,j,prompt))

    prompts = [prompt for _,_,prompt in inputs]

    outputs = model.query(prompts, temperature) if len(prompts) > 0 else []

    choices = [[None for _ in si['groups']] for si in sentences]
    for (i,j,_),response in zip(inputs, outputs):
        options = disambiguated[i]['groups'][j]['options']
        choice = parse_disambiguate_response(response, options)
        choices[i][j] = choice

    return choices