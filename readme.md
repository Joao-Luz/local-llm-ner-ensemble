# Local LLM NER Ensembling

This is the repo for the paper titled "Local LLM Ensembles for Zero-shot Portuguese Named Entity Recognition" approved for CIARP 2025.

VLLM is used as the backend for running the LLMs.

To annotate via an ensemble configuration, run:

```bash
python3 src/annotate.py \
    --unlabeled_data path/to/data_to_annotate.conll \
    --labeled_data path/to/labelled_data.conll \
    --dataset_info path/to/info.json \
    --outdir output/dir/
```

An example of the format for the input files can be seen in `data/harem`

Both the unlabeled and labeled data should be in the BIO/IOB format:

```
John    B-PER
Doe     I-PER
works   O
at      O
Apple   B-ORG

```

The `--dataset_info` is a JSON file with the following schema:

```json
{
    "labels": "List of entity types without BIO prefixes",
    "label_descriptions": "Dictionary mapping each entity label to it's description",
    "label2idx": "Dictionary mapping BIO-formatted labels to integer indices.",
    "idx2label": "Dictionary mapping stringified indices back to BIO labels"
}
```

The `output/dir/` will be populated by the outputs from the stages of the pipeline.

An LLM wrapper is defined at `src/llm.py`. You can call the `LLM` class with a `model_name` with the path to a HuggingFace model or use the shorthands for the LLMs used in this work:

```json
{
    "llama3.1": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen2": "Qwen/Qwen2.5-7B-Instruct",
    "gemma2": "google/gemma-2-9b-it", 
    "phi3:14b": "microsoft/Phi-3-medium-128k-instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2"
}
```
