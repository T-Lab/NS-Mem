# -*- coding: utf-8 -*-
"""Prompt template loader for baseline, NSTF, and evaluation prompts."""

from pathlib import Path
from typing import Optional

PROMPTS_DIR = Path(__file__).parent


def load_prompt(subdir: str, name: str) -> str:
    """Load a prompt template from the specified subdirectory."""
    prompt_path = PROMPTS_DIR / subdir / f"{name}.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def get_prompt_mode(
    mode: Optional[str] = None,
    ablation_mode: Optional[str] = None,
    nstf_available: bool = False
) -> str:
    """Determine which prompt directory to use based on run mode and NSTF availability."""
    if mode:
        if mode == 'nstf_full' and nstf_available:
            return 'nstf'
        elif mode.startswith('ablation_'):
            return 'baseline'
        elif mode == 'baseline':
            return 'baseline'

    if ablation_mode in ['baseline', 'prototype', 'structure']:
        return 'baseline'

    if nstf_available:
        return 'nstf'

    return 'baseline'


def get_system_prompt(
    mode: Optional[str] = None,
    ablation_mode: Optional[str] = None,
    nstf_available: bool = False
) -> str:
    """Get system prompt."""
    prompt_mode = get_prompt_mode(mode, ablation_mode, nstf_available)
    return load_prompt(prompt_mode, 'system')


def get_instruction(
    mode: Optional[str] = None,
    ablation_mode: Optional[str] = None,
    nstf_available: bool = False
) -> str:
    """Get instruction prompt."""
    prompt_mode = get_prompt_mode(mode, ablation_mode, nstf_available)
    return load_prompt(prompt_mode, 'instruction')


def get_evaluation_prompt() -> str:
    """Get evaluation prompt (shared across all modes)."""
    prompt_path = PROMPTS_DIR / "evaluation.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Evaluation prompt file not found: {prompt_path}")

    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read().strip()
