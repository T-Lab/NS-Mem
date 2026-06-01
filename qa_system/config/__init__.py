import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Literal

CONFIG_DIR = Path(__file__).parent

QAMode = Literal["baseline", "nstf_full", "ablation_prototype", "ablation_structure"]


@dataclass
class BaselineConfig:
    """Baseline mode configuration."""
    threshold: float = 0.3
    topk: int = 10
    mem_wise: bool = False


@dataclass
class NSTFConfig:
    """NSTF mode configuration."""
    threshold: float = 0.35
    min_confidence: float = 0.30
    max_procedures: int = 3
    topk_baseline: int = 10
    threshold_baseline: float = 0.3
    include_episodic_evidence: bool = True
    use_reranking: bool = False
    use_dag_paths: bool = True


@dataclass
class QAConfig:
    """QA system configuration."""

    mode: str = "baseline"
    topk: int = 10
    threshold: float = 0.05
    threshold_baseline: float = 0.3
    total_round: int = 5
    batch_size: int = 4
    ablation_mode: Optional[str] = None
    retrieval_strategy: str = 'clip_level'
    nstf_threshold: float = 0.35
    nstf_min_confidence: float = 0.30
    nstf_max_procedures: int = 3
    nstf_include_evidence: bool = True
    nstf_use_reranking: bool = False
    nstf_use_dag_paths: bool = True
    llm_source: str = field(default_factory=lambda: os.environ.get('CONTROL_LLM_SOURCE', 'local'))
    llm_model: str = field(default_factory=lambda: os.environ.get('CONTROL_LLM_MODEL', 'M3-Agent-Control'))
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens: int = 1024
    gpt_model: str = field(default_factory=lambda: os.environ.get('GPT_MODEL', 'gpt-4o-mini'))
    eval_timeout: int = 30
    data_dir: Optional[str] = None
    output_dir: Optional[str] = None

    @classmethod
    def from_json(cls, path: str) -> 'QAConfig':
        """Load config from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if 'llm_source' not in data or data.get('llm_source') is None:
            data['llm_source'] = os.environ.get('CONTROL_LLM_SOURCE', 'local')
        if 'llm_model' not in data or data.get('llm_model') is None:
            data['llm_model'] = os.environ.get('CONTROL_LLM_MODEL', 'M3-Agent-Control')
        if 'gpt_model' not in data or data.get('gpt_model') is None:
            data['gpt_model'] = os.environ.get('GPT_MODEL', 'gpt-4o-mini')

        return cls(**data)

    @classmethod
    def load_default(cls) -> 'QAConfig':
        """Load default config."""
        default_path = CONFIG_DIR / "default.json"
        if default_path.exists():
            return cls.from_json(str(default_path))
        return cls()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'mode': self.mode,
            'topk': self.topk,
            'threshold': self.threshold,
            'threshold_baseline': self.threshold_baseline,
            'total_round': self.total_round,
            'batch_size': self.batch_size,
            'llm_source': self.llm_source,
            'llm_model': self.llm_model,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'max_tokens': self.max_tokens,
            'gpt_model': self.gpt_model,
            'eval_timeout': self.eval_timeout,
            'nstf_threshold': self.nstf_threshold,
            'nstf_min_confidence': self.nstf_min_confidence,
            'nstf_max_procedures': self.nstf_max_procedures,
            'nstf_include_evidence': self.nstf_include_evidence,
            'nstf_use_reranking': self.nstf_use_reranking,
            'nstf_use_dag_paths': self.nstf_use_dag_paths,
        }

    def get_baseline_config(self) -> 'BaselineConfig':
        """Get baseline config subset."""
        return BaselineConfig(
            threshold=self.threshold_baseline,
            topk=self.topk,
        )

    def get_nstf_config(self) -> 'NSTFConfig':
        """Get NSTF config subset."""
        return NSTFConfig(
            threshold=self.nstf_threshold,
            min_confidence=self.nstf_min_confidence,
            max_procedures=self.nstf_max_procedures,
            topk_baseline=self.topk,
            threshold_baseline=self.threshold_baseline,
            include_episodic_evidence=self.nstf_include_evidence,
            use_reranking=self.nstf_use_reranking,
            use_dag_paths=self.nstf_use_dag_paths,
        )

    def save(self, path: str):
        """Save config to JSON."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


MODE_CONFIGS = {
    'baseline': QAConfig(
        mode='baseline',
        threshold_baseline=0.3,
        topk=10,
    ),
    'nstf_full': QAConfig(
        mode='nstf_full',
        nstf_threshold=0.30,
        nstf_min_confidence=0.25,
        nstf_max_procedures=3,
        nstf_include_evidence=True,
        nstf_use_reranking=True,
        nstf_use_dag_paths=True,
    ),
    'ablation_prototype': QAConfig(
        mode='ablation_prototype',
        threshold=0.05,
    ),
    'ablation_structure': QAConfig(
        mode='ablation_structure',
        threshold=0.05,
    ),
}

ABLATION_CONFIGS = {
    'baseline': MODE_CONFIGS['baseline'],
    'prototype': MODE_CONFIGS['ablation_prototype'],
    'structure': MODE_CONFIGS['ablation_structure'],
    'full_nstf': MODE_CONFIGS['nstf_full'],
}


def get_mode_config(mode: str) -> QAConfig:
    """Get config for the specified mode."""
    if mode in MODE_CONFIGS:
        return MODE_CONFIGS[mode]
    if mode in ABLATION_CONFIGS:
        return ABLATION_CONFIGS[mode]
    raise ValueError(f"Unknown mode: {mode}. Available: {list(MODE_CONFIGS.keys())}")


def get_ablation_config(mode: str) -> QAConfig:
    """Get config for the specified ablation mode (legacy compatibility)."""
    if mode not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown ablation mode: {mode}. Available: {list(ABLATION_CONFIGS.keys())}")
    return ABLATION_CONFIGS[mode]
