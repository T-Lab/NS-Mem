import os
import sys
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

from env_setup import NSTF_MODEL_DIR

from dotenv import load_dotenv
load_dotenv(NSTF_MODEL_DIR / '.env', override=False)


def setup_gpu_for_local_llm():
    """Set GPU environment variables for local LLM inference."""
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = os.environ.get('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')

    gpu_devices = os.environ.get('GPU_DEVICES', '0,1,2,3')
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_devices

    gpu_list = [x.strip() for x in gpu_devices.split(',') if x.strip()]
    tensor_parallel_size = len(gpu_list)
    gpu_memory_utilization = float(os.environ.get('GPU_MEMORY_UTILIZATION', '0.85'))
    max_model_len = int(os.environ.get('MAX_MODEL_LEN', '8192'))
    max_num_seqs = int(os.environ.get('MAX_NUM_SEQS', '256'))
    enforce_eager = os.environ.get('ENFORCE_EAGER', 'false').lower() == 'true'

    return {
        'tensor_parallel_size': tensor_parallel_size,
        'gpu_memory_utilization': gpu_memory_utilization,
        'max_model_len': max_model_len,
        'max_num_seqs': max_num_seqs,
        'enforce_eager': enforce_eager,
    }


class LLMClient:
    """Unified LLM client wrapper for cloud (OpenAI-compatible) and local (vLLM) models."""

    def __init__(
        self,
        source: str = "cloud",
        model: str = None,
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_tokens: int = 4096,
        **kwargs
    ):
        self.source = source
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

        if source == "cloud":
            self._init_cloud(model)
        elif source == "local":
            self._init_local(model, **kwargs)
        else:
            raise ValueError(f"Unknown LLM source: {source}")

    def _init_cloud(self, model: str = None):
        """Initialize cloud client."""
        from openai import OpenAI

        self.model = model or os.getenv("CONTROL_LLM_MODEL", "gemini-2.5-flash")
        self.client = OpenAI(
            api_key=os.getenv("CONTROL_LLM_API_KEY"),
            base_url=os.getenv("CONTROL_LLM_BASE_URL"),
        )

    def _init_local(self, model: str = None, **kwargs):
        """Initialize local vLLM engine."""
        gpu_config = setup_gpu_for_local_llm()

        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        self.model_path = str(NSTF_MODEL_DIR / "models" / "M3-Agent-Control")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Local model not found: {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        self.model = LLM(
            model=self.model_path,
            tensor_parallel_size=gpu_config['tensor_parallel_size'],
            gpu_memory_utilization=gpu_config['gpu_memory_utilization'],
            max_model_len=gpu_config['max_model_len'],
            max_num_seqs=gpu_config['max_num_seqs'],
            disable_log_stats=True,
            trust_remote_code=True,
            disable_custom_all_reduce=True,
            enforce_eager=gpu_config['enforce_eager'],
            dtype="bfloat16"
        )

        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=20,
            max_tokens=self.max_tokens
        )

    def generate(
        self,
        messages: List[Dict[str, str]],
        timeout: int = 120,
    ) -> str:
        """Generate a response from the LLM."""
        if self.source == "cloud":
            return self._generate_cloud(messages, timeout)
        else:
            return self._generate_local(messages)

    def _generate_cloud(self, messages: List[Dict], timeout: int) -> str:
        """Cloud API generation."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                timeout=timeout,
            )
            return response.choices[0].message.content
        except Exception:
            return ""

    def _generate_local(self, messages: List[Dict]) -> str:
        """Local vLLM generation."""
        try:
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=True
            )

            max_input_len = int(os.getenv('MAX_MODEL_LEN', '8192')) - self.max_tokens
            if len(input_ids) > max_input_len:
                return "[SKIPPED: Input too long]"

            outputs = self.model.generate(
                prompts=[{"prompt_token_ids": input_ids}],
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
            return outputs[0].outputs[0].text
        except Exception:
            return ""

    def batch_generate(
        self,
        batch_messages: List[List[Dict]],
        timeout: int = 120,
    ) -> List[str]:
        """Batch generation (cloud: sequential, local: parallel)."""
        if self.source == "cloud":
            results = []
            for messages in batch_messages:
                results.append(self._generate_cloud(messages, timeout))
            return results
        else:
            return self._batch_generate_local(batch_messages)

    def _batch_generate_local(self, batch_messages: List[List[Dict]]) -> List[str]:
        """Local batch generation via vLLM."""
        prompts = []
        valid_indices = []
        max_input_len = int(os.getenv('MAX_MODEL_LEN', '8192')) - self.max_tokens

        for i, messages in enumerate(batch_messages):
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=True
            )
            if len(input_ids) <= max_input_len:
                prompts.append({"prompt_token_ids": input_ids})
                valid_indices.append(i)

        if not prompts:
            return ["[SKIPPED]"] * len(batch_messages)

        outputs = self.model.generate(
            prompts=prompts,
            sampling_params=self.sampling_params,
            use_tqdm=False,
        )

        results = ["[SKIPPED]"] * len(batch_messages)
        for idx, output in zip(valid_indices, outputs):
            results[idx] = output.outputs[0].text

        return results
