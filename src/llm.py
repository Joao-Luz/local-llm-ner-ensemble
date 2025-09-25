import vllm

import contextlib
import gc
import ray
import torch
from vllm.distributed import destroy_model_parallel, destroy_distributed_environment

class LLM():
    def __init__(self, model_name=None):
        self.model = None
        self.name = ''

        if model_name is not None:
            self.name = model_name
            # self.model = self.load_model(model_name)

    def _delete_model(self):
        if self.model is None:
            return

        destroy_model_parallel()
        destroy_distributed_environment()

        del self.model.llm_engine.model_executor
        del self.model

        with contextlib.suppress(AssertionError):
            torch.distributed.destroy_process_group()

        gc.collect()
        torch.cuda.empty_cache()
        ray.shutdown()

        self.model = None
        self.name = ''

    def __del__(self):
        self._delete_model()

    def _name_mapping(self, name):
        mapping = {
            'llama3.1': 'meta-llama/Llama-3.1-8B-Instruct',
            'qwen2': 'Qwen/Qwen2.5-7B-Instruct',
            'gemma2': 'google/gemma-2-9b-it', 
            'phi3:14b': 'microsoft/Phi-3-medium-128k-instruct',
            'mistral': 'mistralai/Mistral-7B-Instruct-v0.2'
        }

        return mapping[name] if name in mapping else name

    def load_model(self, model_name):
        self._delete_model()

        model_path = self._name_mapping(model_name)

        self.name = model_name

        if model_name == 'phi3:14b':
            model = vllm.LLM(model=model_path, trust_remote_code=True,
                             max_model_len=2048,
                             gpu_memory_utilization=0.9,
                             )
        else:
            model = vllm.LLM(model=model_path, trust_remote_code=True,
                             max_model_len=2048,
                             gpu_memory_utilization=0.9,
                             )
    
        self.model = model
    
    def query(self, prompts, temperature):
        if self.model is None and self.name:
            self.load_model(self.name)

        chats = [[{'role': 'user', 'content': prompt}] for prompt in prompts]
        sampling_params = vllm.SamplingParams(temperature=temperature, max_tokens=1024)

        outputs = self.model.chat(chats, sampling_params)
        generated_texts = [output.outputs[0].text for output in outputs]

        return generated_texts