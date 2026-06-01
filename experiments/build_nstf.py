#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).parent.resolve()
NSTF_MODEL_DIR = EXPERIMENTS_DIR.parent

os.chdir(NSTF_MODEL_DIR)

sys.path.insert(0, str(NSTF_MODEL_DIR))
from env_setup import setup_all
setup_all()


def get_video_list_from_annotations(dataset: str, data_dir: Path) -> list:
    ann_dir = data_dir / "annotations" / dataset
    if ann_dir.exists():
        return sorted([f.stem for f in ann_dir.glob("*.json")])

    ann_file = data_dir / "annotations" / f"{dataset}.json"
    if ann_file.exists():
        with open(ann_file, 'r') as f:
            data = json.load(f)
        return list(data.keys())

    return []


def main():
    parser = argparse.ArgumentParser(description='Build NSTF graphs')
    parser.add_argument('--video-list', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='robot',
                        choices=['robot', 'web'])
    parser.add_argument('--mode', type=str, default='incremental',
                        choices=['static', 'incremental'])
    parser.add_argument('--max-procedures', type=int, default=5)
    parser.add_argument('--force', '-f', action='store_true')
    parser.add_argument('--videos', type=str, nargs='+', default=None)
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    data_dir = NSTF_MODEL_DIR / 'data'

    if args.videos:
        videos = args.videos
    elif args.video_list:
        video_list_path = NSTF_MODEL_DIR / args.video_list
        with open(video_list_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        videos = data.get('videos', data) if isinstance(data, dict) else data
    else:
        videos = get_video_list_from_annotations(args.dataset, data_dir)

    if not videos:
        return 1

    output_dir = data_dir / 'nstf_graphs' / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = '_nstf.pkl'

    if args.force:
        to_process = videos
    else:
        existing = set()
        for f in output_dir.glob(f'*{suffix}'):
            video_name = f.stem.replace('_nstf_incremental', '').replace('_nstf', '')
            existing.add(video_name)
        to_process = [v for v in videos if v not in existing]

    if not to_process:
        return 0

    if args.mode == 'incremental':
        from nstf_builder import IncrementalNSTFBuilder
        builder = IncrementalNSTFBuilder(
            data_dir=str(data_dir),
            output_dir=str(data_dir / 'nstf_graphs'),
            debug=args.debug,
        )
        builder.build_batch(to_process, dataset=args.dataset)
    else:
        from nstf_builder import NSTFBuilder
        builder = NSTFBuilder(
            data_dir=str(data_dir),
            output_dir=str(data_dir / 'nstf_graphs'),
            debug=args.debug,
        )
        builder.build_batch(to_process, dataset=args.dataset, max_procedures=args.max_procedures)

    return 0


if __name__ == '__main__':
    sys.exit(main())
