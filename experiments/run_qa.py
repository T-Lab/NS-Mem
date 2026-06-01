#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).parent.resolve()
NSTF_MODEL_DIR = EXPERIMENTS_DIR.parent

os.chdir(NSTF_MODEL_DIR)

sys.path.insert(0, str(NSTF_MODEL_DIR))
from env_setup import setup_all
setup_all()

from qa_system import QARunner
from qa_system.config import QAConfig, get_ablation_config


def resolve_path(path_str: str) -> str:
    if path_str is None:
        return None
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    return str(NSTF_MODEL_DIR / p)


def main():
    parser = argparse.ArgumentParser(description='Run NS-Mem QA experiments')

    parser.add_argument('--dataset', type=str, default='web',
                        choices=['robot', 'web'])
    parser.add_argument('--data-file', type=str, default=None)
    parser.add_argument('--video-list', type=str, default=None)
    parser.add_argument('--videos', type=str, default=None)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--query-type-file', type=str, default=None)
    parser.add_argument('--ablation', type=str, default=None,
                        choices=['baseline', 'prototype', 'structure', 'full_nstf', 'nstf_level', 'nstf_node'])
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--llm-source', type=str, default=None,
                        choices=['cloud', 'local'])
    parser.add_argument('--llm-model', type=str, default=None)

    args = parser.parse_args()

    if args.ablation:
        config = get_ablation_config(args.ablation)
    else:
        config = QAConfig.load_default()

    if args.llm_source is not None:
        config.llm_source = args.llm_source
    if args.llm_model is not None:
        config.llm_model = args.llm_model

    data_file = resolve_path(args.data_file)
    video_list_file = resolve_path(args.video_list)
    output_file = resolve_path(args.output)
    query_type_file = resolve_path(args.query_type_file)

    runner = QARunner(config=config)

    video_names = None
    if args.videos:
        video_names = [v.strip() for v in args.videos.split(',')]

    result = runner.run(
        dataset=args.dataset,
        video_names=video_names,
        video_list_file=video_list_file,
        data_file=data_file,
        query_type_file=query_type_file,
        limit=args.limit,
        output_file=output_file,
        resume=not args.no_resume,
        force=args.force,
    )

    return 0 if result['accuracy'] > 0 else 1


if __name__ == '__main__':
    sys.exit(main())
