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
import logging
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(BASE_DIR, "configs", "processing_config.json")
processing_config = json.load(open(config_path))
logging_level = processing_config["logging"]
model = processing_config["model"]

if processing_config.get("train", False):
    logging.basicConfig(level=logging.CRITICAL)
else:
    logging.basicConfig(
        level=logging.DEBUG if logging_level == "DETAIL" else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(processing_config["log_dir"], "mmagent.log"))
        ]
    )

logging.getLogger('moviepy').setLevel(logging.ERROR)
logging.getLogger('moviepy.video.io.VideoFileClip').setLevel(logging.ERROR)
logging.getLogger('moviepy.audio.io.AudioFileClip').setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("httpcore").setLevel(logging.CRITICAL)

from . import retrieve
from . import face_processing
from . import memory_processing
try:
    if model == "qwen2.5-omni":
        from . import memory_processing_qwen
except:
    pass
from . import prompts
from . import videograph
from . import voice_processing
from . import utils

__all__ = [
    "retrieve",
    "face_processing",
    "memory_processing",
    "memory_processing_qwen" if model == "qwen2.5-omni" else None,
    "prompts",
    "videograph",
    "voice_processing",
    "utils",
]