# Copyright (2025) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import json
import subprocess
from datetime import datetime
from moviepy import *
import re
import ast
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
from tqdm import tqdm
import struct
import pickle
import shutil
import logging
from .chat_api import parallel_get_whisper

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
processing_config = json.load(open(os.path.join(BASE_DIR, "configs", "processing_config.json")))
memory_config = json.load(open(os.path.join(BASE_DIR, "configs", "memory_config.json")))
TEMP_DIR = "temp"

def get_video_paths(video_url, task):
    """Generate video and segment paths from a URL and task identifier."""
    video_name = video_url.split("/")[-1].split(".")[0].split("_")[3]
    segment_name = video_url.split("/")[-1].split(".")[0]
    video_save_path = os.path.join(f"output/{task}", video_name)
    segment_save_path = os.path.join(video_save_path, segment_name)
    return video_save_path, segment_save_path

def get_video_names(path):
    """Extract unique video names from filenames in a directory."""
    files = os.listdir(path)
    video_names = [file.split("_")[3] for file in files]
    video_names = list(set(video_names))
    return video_names

def get_files_by_name(base_path, video_name, video_config):
    """Get sorted video files matching the specified name and config prefix."""
    files = os.listdir(base_path)
    prefix = video_config["resolution"] + "_" + video_config["clip_size"] + "_" + video_config["clip_duration"] + "_" + video_name
    video_files = [
        file
        for file in files
        if (file.startswith(prefix))
    ]

    video_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    return video_files

def get_files_by_title(base_path, title, video_config):
    """Get video files matching the MD5-hashed title prefix."""
    # Calculate the md5 hash of the title, first 8 characters
    command = f'echo "{title}" | md5sum | cut -c1-8'
    result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
    title_hash = result.stdout.strip()

    return title_hash, get_files_by_name(base_path, title_hash, video_config)

def generate_test_file_name(sample, task):
    """Generate a timestamped test file name from sample and task."""
    if sample.endswith(".mp4"):
        sample = sample.split("/")[-1].split(".")[0]
    date = datetime.now().strftime("%Y%m%d")
    return f"{date}_{sample}_{task}"


def generate_audio_files(video, video_config, base_path_video, base_path_audio):
    """Extract audio from video files and save as MP3."""
    video_files = get_files_by_name(base_path_video, video, video_config)
    for video_file in video_files:
        input_path = os.path.join(base_path_video, video_file)
        output_path = os.path.join(base_path_audio, video_file.replace(".mp4", ".mp3"))

        video_clip = VideoFileClip(input_path)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(output_path)
        audio_clip.close()
        video_clip.close()

        logger.info(f"Audio successfully extracted and saved as: {output_path}")

def generate_transcripts(video, video_config, base_path):
    """Generate transcripts from audio files using the Whisper model."""
    audio_files = get_files_by_name(base_path, video, video_config)
    audio_paths = [os.path.join(base_path, audio_file) for audio_file in audio_files]
    transcripts = parallel_get_whisper("whisper", audio_paths)

    for i, transcript in enumerate(transcripts):
        with open(f"../data/transcripts/{audio_files[i].split('.')[0]}.txt", "w", encoding="utf-8") as f:
            f.write(transcript)

def load_subtitle(subtitle_path):
    """Load and parse an SRT subtitle file, extracting only dialogue lines."""
    with open(subtitle_path, "r") as f:
        lines = f.readlines()
    # Keep only every 4th line starting from index 2 (the dialogue lines)
    lines = [line.strip("\n") for i, line in enumerate(lines) if i % 4 == 2]
    return " ".join(lines)

def load_transcript(transcript_path):
    """Load transcript text from a file."""
    with open(transcript_path, "r") as f:
        transcript = f.read()
    return transcript

def refine_json_str(invalid_json):
    """Clean a JSON string by removing markdown code block wrappers."""
    # invalid_json = invalid_json.strip()
    # if invalid_json.startswith("```json"):
    #     invalid_json = invalid_json[7:].strip()
    # if invalid_json.endswith("```"):
    #     invalid_json = invalid_json[:-3].strip()

    # fixed_json = re.sub(r"'", '"', invalid_json)
    fixed_json = invalid_json.strip("```json").strip("```python").strip("```").strip()

    # # Fix keys without double quotes
    # fixed_json = re.sub(r'(?<=\{|,)\s*([a-zA-Z0-9_]+)\s*:', r'"\1":', fixed_json)

    # # Auto-complete missing braces and brackets
    # stack = []
    # for char in fixed_json:
    #     if char in '{[':
    #         stack.append(char)
    #     elif char in '}]':
    #         if stack and ((char == '}' and stack[-1] == '{') or (char == ']' and stack[-1] == '[')):
    #             stack.pop()

    # # Complete missing brackets
    # while stack:
    #     last = stack.pop()
    #     if last == '{':
    #         fixed_json += '}'
    #     elif last == '[':
    #         fixed_json += ']'

    # # Check if quotes are balanced
    # if fixed_json.count('"') % 2 != 0:
    #     fixed_json += '"'

    return fixed_json

def validate_and_fix_json(invalid_json):
    fixed_json = refine_json_str(invalid_json)
    try:
        return json.loads(fixed_json)
    except json.JSONDecodeError as e:
        logger.error(f"Still unable to fix: {e}")
        logger.error(invalid_json)
        return None

def validate_and_fix_python_list(invalid_list_string):
    """Validate and parse a Python list string, stripping markdown wrappers."""
    try:
        s = invalid_list_string.strip("```json").strip("```python").strip("```").strip()
        result = ast.literal_eval(s)
        if isinstance(result, list):
            return result
        else:
            raise ValueError("Input string is not a list")
    except (SyntaxError, ValueError) as e:
        logger.error(f"Parsing error: {e}")
        logger.error(invalid_list_string)
        return None

def plot_cosine_similarity_distribution(embeddings1, embeddings2, save_path=None, max_num=2000):
    embeddings1 = embeddings1[np.random.choice(len(embeddings1), min(max_num, len(embeddings1)), replace=False)]
    embeddings2 = embeddings2[np.random.choice(len(embeddings2), min(max_num, len(embeddings2)), replace=False)]

    sim_scores = cosine_similarity(embeddings1, embeddings2).flatten()

    plt.figure(figsize=(8, 5))
    plt.hist(sim_scores, bins=30, color='skyblue', edgecolor='black')
    plt.title('Cross-Group Cosine Similarity Distribution')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    if save_path:
        temp_path = os.path.join(TEMP_DIR, os.path.basename(save_path))
        plt.savefig(temp_path, dpi=300, bbox_inches='tight')
        plt.close()
        shutil.move(temp_path, save_path)
    else:
        plt.show()

def plot_value_distribution(values, save_path=None, title='Value Distribution', bins=30):
    """Plot and optionally save a histogram of the given values."""
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    if save_path:
        temp_path = os.path.join(TEMP_DIR, os.path.basename(save_path))
        plt.savefig(temp_path, dpi=300, bbox_inches='tight')
        plt.close()
        shutil.move(temp_path, save_path)
    else:
        plt.show()


def normalize_embedding(embedding):
    """Normalize an embedding vector to unit length."""
    format_string = 'f' * (len(embedding) // struct.calcsize('f'))
    emb = np.array(struct.unpack(format_string, embedding))
    norm = np.linalg.norm(emb)
    return (emb / norm).tolist() if norm > 0 else emb.tolist()

def generate_file_name(video_path):
    return f"{video_path.split('/')[-1].split('.')[0].replace(' ', '-')}_{processing_config['interval_seconds']}_{processing_config['fps']}_{processing_config['segment_limit']}_{memory_config['max_img_embeddings']}_{memory_config['max_audio_embeddings']}_{memory_config['img_matching_threshold']}_{memory_config['audio_matching_threshold']}"

def get_video_prefix(clip_id, video_path):
    pass

def save_video_graph(video_graph, video_path, save_dir, file_name=None):
    """Save a video graph object to a pickle file."""
    if not file_name:
        file_name = generate_file_name(video_path) + ".pkl"
    temp_save_dir = "data/mems"
    os.makedirs(temp_save_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file_name)
    temp_save_path = os.path.join(temp_save_dir, file_name)
    with open(temp_save_path, "wb") as f:
        logger.info(f"Saving video graph to {temp_save_path}")
        pickle.dump(video_graph, f)
    logger.info(f"Moving video graph to {save_path}")
    shutil.move(temp_save_path, save_path)

def load_video_graph(video_graph_path):
    """Load a video graph object from a pickle file."""
    if not os.path.exists(video_graph_path):
        logger.warning("Video graph not found")
        return None
    with open(video_graph_path, "rb") as f:
        logger.info(f"Loading video graph from {video_graph_path}")
        return pickle.load(f)
