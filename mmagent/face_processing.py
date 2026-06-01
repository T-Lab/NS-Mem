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
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import json
import os
import logging
import gc
from insightface.app import FaceAnalysis
from mmagent.src.face_extraction import extract_faces
from mmagent.src.face_clustering import cluster_faces
from mmagent.utils.video_processing import process_video_clip

INSIGHTFACE_HOME = os.environ.get("INSIGHTFACE_HOME", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "insightface"))
os.makedirs(INSIGHTFACE_HOME, exist_ok=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(BASE_DIR, "configs", "processing_config.json")
processing_config = json.load(open(config_path))

GPU_CONFIG = processing_config.get('gpu_config', {
    'gpu_id': 'auto',
    'force_cpu': False,
    'face_detection_size': [480, 480],
    'auto_fallback_to_cpu': True,
    'max_retry_on_oom': 2
})

def clear_gpu_memory():
    """Clear GPU memory."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    except:
        gc.collect()

face_app = None
face_app_ctx_id = None

def get_face_app():
    """Get face analysis model (lazy initialization)."""
    global face_app, face_app_ctx_id

    if face_app is not None:
        return face_app

    import onnxruntime as ort
    logger = logging.getLogger(__name__)

    available_providers = ort.get_available_providers()
    logger.info(f"Available ONNX Runtime providers: {available_providers}")

    force_cpu = os.environ.get('INSIGHTFACE_FORCE_CPU', str(GPU_CONFIG.get('force_cpu', False))).lower() == 'true'

    if force_cpu:
        logger.info("Forced CPU mode (INSIGHTFACE_FORCE_CPU=true)")
        face_app = FaceAnalysis(name="buffalo_l", root=INSIGHTFACE_HOME, providers=['CPUExecutionProvider'])
        face_app.prepare(ctx_id=-1, det_size=(480, 480))
        face_app_ctx_id = -1
        logger.info("InsightFace initialized (CPU mode)")
        return face_app

    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    gpu_ids = [int(x.strip()) for x in cuda_visible.split(',') if x.strip()]
    num_gpus = len(gpu_ids)

    config_gpu_id = GPU_CONFIG.get('gpu_id', 'auto')
    if config_gpu_id == 'auto':
        ctx_id = 0
        logger.info("InsightFace config: auto GPU mode, using mapped GPU 0")
    elif isinstance(config_gpu_id, int):
        if config_gpu_id == -1:
            logger.info("Config specifies CPU mode")
            face_app = FaceAnalysis(name="buffalo_l", root=INSIGHTFACE_HOME, providers=['CPUExecutionProvider'])
            det_size = tuple(GPU_CONFIG.get('face_detection_size', [480, 480]))
            face_app.prepare(ctx_id=-1, det_size=det_size)
            face_app_ctx_id = -1
            logger.info(f"InsightFace initialized (CPU mode, det_size={det_size})")
            return face_app
        elif 0 <= config_gpu_id < num_gpus:
            ctx_id = config_gpu_id
            logger.info(f"Using configured GPU {ctx_id} (physical: {gpu_ids[ctx_id] if ctx_id < len(gpu_ids) else 'N/A'})")
        else:
            logger.warning(f"Configured GPU ID {config_gpu_id} out of range, using GPU 0")
            ctx_id = 0
    else:
        ctx_id = 0
        logger.warning(f"Invalid GPU config '{config_gpu_id}', using GPU 0")

    logger.info(f"InsightFace config: {num_gpus} GPUs available (CUDA_VISIBLE_DEVICES={cuda_visible})")

    max_retry = GPU_CONFIG.get('max_retry_on_oom', 2)
    for attempt in range(max_retry):
        try:

            face_app = FaceAnalysis(
                name="buffalo_l",
                root=INSIGHTFACE_HOME,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )

            base_size = GPU_CONFIG.get('face_detection_size', [480, 480])
            if attempt == 0:
                det_size = tuple(base_size)
            elif attempt == 1:
                det_size = (max(base_size[0] - 160, 320), max(base_size[1] - 160, 320))
            else:
                det_size = (320, 320)

            logger.info(f"Initializing InsightFace (GPU {ctx_id}, det_size={det_size}, attempt {attempt+1}/{max_retry})...")
            face_app.prepare(ctx_id=ctx_id, det_size=det_size)

            face_app_ctx_id = ctx_id
            logger.info(f"InsightFace initialized on GPU {ctx_id} (physical: {gpu_ids[0]}, det_size={det_size})")
            return face_app

        except Exception as e:
            logger.warning(f"InsightFace GPU init failed (attempt {attempt+1}/{max_retry}): {e}")
            if attempt == max_retry - 1:
                break

    if GPU_CONFIG.get('auto_fallback_to_cpu', True):
        logger.warning("GPU init failed, falling back to CPU mode")
        face_app = FaceAnalysis(name="buffalo_l", root=INSIGHTFACE_HOME, providers=['CPUExecutionProvider'])
        det_size = tuple(GPU_CONFIG.get('face_detection_size', [480, 480]))
        face_app.prepare(ctx_id=-1, det_size=det_size)
        face_app_ctx_id = -1
        logger.info(f"InsightFace initialized (CPU mode, det_size={det_size})")
        return face_app
    else:
        raise RuntimeError("GPU init failed and CPU fallback is disabled (auto_fallback_to_cpu=false)")

cluster_size = processing_config["cluster_size"]
logger = logging.getLogger(__name__)

class Face:
    def __init__(self, frame_id, bounding_box, face_emb, cluster_id, extra_data):
        self.frame_id = frame_id
        self.bounding_box = bounding_box
        self.face_emb = face_emb
        self.cluster_id = cluster_id
        self.extra_data = extra_data

def get_face(frames):
    """Extract faces from frames (lazy-initializes face_app)."""
    app = get_face_app()
    extracted_faces = extract_faces(app, frames)
    faces = [Face(frame_id=f['frame_id'], bounding_box=f['bounding_box'], face_emb=f['face_emb'], cluster_id=f['cluster_id'], extra_data=f['extra_data']) for f in extracted_faces]
    return faces

def cluster_face(faces):
    faces_json = [{'frame_id': f.frame_id, 'bounding_box': f.bounding_box, 'face_emb': f.face_emb, 'cluster_id': f.cluster_id, 'extra_data': f.extra_data} for f in faces]
    clustered_faces = cluster_faces(faces_json, 20, 0.5)
    faces = [Face(frame_id=f['frame_id'], bounding_box=f['bounding_box'], face_emb=f['face_emb'], cluster_id=f['cluster_id'], extra_data=f['extra_data']) for f in clustered_faces]
    return faces

def process_faces(video_graph, base64_frames, save_path, preprocessing=[]):
    """Detect, cluster, and track faces in video frames, updating the video graph."""
    logger.info("Starting face detection")
    logger.info(f"Input frames: {len(base64_frames)}, save_path: {save_path}")

    batch_size = max(min(len(base64_frames) // cluster_size, 16), 4)
    logger.info(f"Batch size: {batch_size}")

    def _process_batch(params):
        """Process a batch of frames and return detected faces with adjusted frame IDs."""
        frames = params[0]
        offset = params[1]
        faces = get_face(frames)
        for face in faces:
            face.frame_id += offset
        return faces

    def get_embeddings(base64_frames, batch_size):
        num_batches = (len(base64_frames) + batch_size - 1) // batch_size
        batched_frames = [
            (base64_frames[i * batch_size : (i + 1) * batch_size], i * batch_size)
            for i in range(num_batches)
        ]

        faces = []

        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
        num_gpus = len(cuda_visible.split(','))
        max_workers = min(num_batches, num_gpus * 4 if num_gpus > 0 else 4)

        logger.info(f"Batch config: {num_batches} batches, batch_size={batch_size}, max_workers={max_workers}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for batch_faces in tqdm(
                executor.map(_process_batch, batched_frames), total=num_batches, desc="Processing face batches"
            ):
                faces.extend(batch_faces)

        faces = cluster_face(faces)
        return faces

    def establish_mapping(faces, key="cluster_id", filter=None):
        mapping = {}
        filtered_count = 0
        for face in faces:
            if key not in face.keys():
                raise ValueError(f"key {key} not found in faces")
            if filter and not filter(face):
                filtered_count += 1
                continue
            id = face[key]
            if id not in mapping:
                mapping[id] = []
            mapping[id].append(face)

        if filtered_count > 0:
            logger.info(f"Filtered {filtered_count}/{len(faces)} faces "
                       f"(detection_threshold={processing_config['face_detection_score_threshold']}, "
                       f"quality_threshold={processing_config['face_quality_score_threshold']})")
        max_faces = processing_config["max_faces_per_character"]
        for id in mapping:
            mapping[id] = sorted(
                mapping[id],
                key=lambda x: (
                    float(x["extra_data"]["face_detection_score"]),
                    float(x["extra_data"]["face_quality_score"]),
                ),
                reverse=True,
            )[:max_faces]
        return mapping

    def filter_score_based(face):
        dthresh = processing_config["face_detection_score_threshold"]
        qthresh = processing_config["face_quality_score_threshold"]
        dscore = float(face["extra_data"]["face_detection_score"])
        qscore = float(face["extra_data"]["face_quality_score"])
        passed = dscore > dthresh and qscore > qthresh

        return passed

    def update_videograph(video_graph, tempid2faces):
        id2faces = {}
        for tempid, faces in tempid2faces.items():
            if tempid == -1:
                continue
            if len(faces) == 0:
                continue
            face_info = {
                "embeddings": [face["face_emb"] for face in faces],
                "contents": [face["extra_data"]["face_base64"] for face in faces],
            }
            matched_nodes = video_graph.search_img_nodes(face_info)
            if len(matched_nodes) > 0:
                matched_node = matched_nodes[0][0]
                video_graph.update_node(matched_node, face_info)
                for face in faces:
                    face["matched_node"] = matched_node
            else:
                matched_node = video_graph.add_img_node(face_info)
                for face in faces:
                    face["matched_node"] = matched_node
            if matched_node not in id2faces:
                id2faces[matched_node] = []
            id2faces[matched_node].extend(faces)

        max_faces = processing_config["max_faces_per_character"]
        for id, faces in id2faces.items():
            id2faces[id] = sorted(
                faces,
                key=lambda x: (
                    float(x["extra_data"]["face_detection_score"]),
                    float(x["extra_data"]["face_quality_score"]),
                ),
                reverse=True
            )[:max_faces]

        return id2faces

    try:
        logger.info(f"Trying to load cached results: {save_path}")
        with open(save_path, "r") as f:
            faces_json = json.load(f)
        logger.info(f"Loaded cached results, {len(faces_json)} faces found")
    except Exception as e:
        logger.info(f"No cached results ({e}), running face detection")

        try:
            faces = get_embeddings(base64_frames, batch_size)
            logger.info(f"Face detection complete, {len(faces)} faces detected")

            faces_json = [
                {
                    "frame_id": face.frame_id,
                    "bounding_box": face.bounding_box,
                    "face_emb": face.face_emb,
                    "cluster_id": int(face.cluster_id),
                    "extra_data": face.extra_data,
                }
                for face in faces
            ]

            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            with open(save_path, "w") as f:
                json.dump(faces_json, f)
            logger.info(f"Intermediate results saved to: {save_path}")

            clear_gpu_memory()
        except Exception as detection_error:
            logger.error(f"Face detection error: {detection_error}")
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            return {}

    if "face" in preprocessing:
        logger.info("Preprocessing mode, returning early")
        return

    if len(faces_json) == 0:
        logger.info("No faces detected, returning empty dict")
        return {}

    tempid2faces = establish_mapping(faces_json, key="cluster_id", filter=filter_score_based)
    logger.info(f"Mapping complete, {len(tempid2faces)} clusters")

    if len(tempid2faces) == 0:
        logger.info(f"No faces passed filtering (detection={processing_config['face_detection_score_threshold']}, "
                   f"quality={processing_config['face_quality_score_threshold']})")
        return {}

    if video_graph is None:
        logger.info("video_graph=None, skipping graph update, returning cluster results")

        id2faces = {}
        for cluster_id, faces in tempid2faces.items():
            if cluster_id == -1:
                continue
            if len(faces) == 0:
                continue
            max_faces = processing_config["max_faces_per_character"]
            sorted_faces = sorted(
                faces,
                key=lambda x: (
                    float(x["extra_data"]["face_detection_score"]),
                    float(x["extra_data"]["face_quality_score"]),
                ),
                reverse=True
            )[:max_faces]
            id2faces[f"character_{cluster_id}"] = sorted_faces

        logger.info(f"Returning {len(id2faces)} clustered characters")

        clear_gpu_memory()

        return id2faces

    id2faces = update_videograph(video_graph, tempid2faces)
    logger.info(f"VideoGraph updated, returning {len(id2faces)} characters")

    clear_gpu_memory()

    return id2faces
