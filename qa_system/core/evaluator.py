import os
import sys
import time
from typing import Tuple, Optional
from pathlib import Path

from env_setup import setup_paths
setup_paths()

import openai
from mmagent.utils.chat_api import generate_messages

from ..prompts import get_evaluation_prompt


class Evaluator:
    """GPT-based answer correctness evaluator."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: int = 30,
        max_retries: int = 3,
    ):
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            self.client = None
        else:
            self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

        self.prompt_template = get_evaluation_prompt()

    def _get_response(self, messages: list) -> Tuple[str, int]:
        """Get GPT response."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            timeout=self.timeout,
            max_tokens=128,
        )
        return response.choices[0].message.content, response.usage.total_tokens

    def _get_response_with_retry(self, messages: list) -> Tuple[str, int]:
        """GPT call with retry."""
        for i in range(self.max_retries):
            try:
                return self._get_response(messages)
            except Exception as e:
                time.sleep(5 * (i + 1))
        raise Exception(f"Evaluation failed after {self.max_retries} retries")

    def evaluate(
        self,
        question: str,
        prediction: str,
        ground_truth: str,
    ) -> bool:
        """Evaluate whether the predicted answer is correct."""
        if not prediction or prediction.strip() == "":
            return False

        if self.client is None:
            return self._simple_match(prediction, ground_truth)

        try:
            prompt_text = self.prompt_template.format(
                question=question,
                ground_truth_answer=ground_truth,
                agent_answer=prediction,
            )

            messages = generate_messages([{"type": "text", "content": prompt_text}])
            response, _ = self._get_response_with_retry(messages)

            return "yes" in response.lower()

        except Exception:
            return self._simple_match(prediction, ground_truth)

    def _simple_match(self, prediction: str, ground_truth: str) -> bool:
        """Simple string matching fallback."""
        pred = prediction.lower().strip()
        gt = ground_truth.lower().strip()

        if gt in pred or pred in gt:
            return True

        if gt in ['yes', 'no', 'yes.', 'no.']:
            return gt.replace('.', '') == pred.replace('.', '')

        return False

    def batch_evaluate(
        self,
        results: list,
        question_key: str = 'question',
        prediction_key: str = 'prediction',
        ground_truth_key: str = 'ground_truth',
        delay: float = 0.5,
    ) -> list:
        """Batch evaluation."""
        for r in results:
            r['gpt_eval'] = self.evaluate(
                r[question_key],
                r.get(prediction_key, ''),
                r[ground_truth_key],
            )
            time.sleep(delay)

        return results
