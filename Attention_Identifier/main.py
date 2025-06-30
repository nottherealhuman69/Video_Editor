import cv2
import numpy as np
import requests
import json
import re
from ultralytics import YOLO

# Load YOLO model once
yolo_model = YOLO("yolov8n.pt")

# ======= Emotion + Weight Fetch via HuggingFace Mistral API =======
from dotenv import load_dotenv
import os

# Load variables from .env into the environment
load_dotenv()

# Access the API key
api_key = os.getenv("API_KEY")

MISTRAL_ENDPOINT = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"


headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

def extract_first_json(text):
    """
    Extract the first valid JSON object from raw LLM text output.
    """
    try:
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception as e:
        print("⚠️ JSON parsing error:", e)
    return None

def get_llm_weights_and_emotion_score(user_description, user_emotion):
    prompt = f"""
<|user|>
You are a video analysis AI. A user uploaded a video described as:

"{user_description}"

The intended emotion is: "{user_emotion}"

Return ONLY a valid JSON object with the following keys:
- "emotion_score": float (0.0 to 1.0)
- "emotion_weight": float
- "saliency_weight": float
- "object_weight": float
- "motion_weight": float

Make sure all weights sum to approximately 1.0.

Example:
{{
  "emotion_score": 0.83,
  "emotion_weight": 0.35,
  "saliency_weight": 0.30,
  "object_weight": 0.20,
  "motion_weight": 0.15
}}
<|assistant|>
"""

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 256,
            "return_full_text": False,
            "do_sample": False
        }
    }

    try:
        response = requests.post(MISTRAL_ENDPOINT, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        generated = response.json()[0]["generated_text"]
        parsed = extract_first_json(generated)
        if parsed:
            return parsed
        else:
            print("⚠️ Could not parse JSON from LLM response. Falling back.")
    except Exception as e:
        print("⚠️ Mixtral API Error:", e)

    # Fallback default
    return {
        "emotion_score": 0.5,
        "emotion_weight": 0.25,
        "saliency_weight": 0.25,
        "object_weight": 0.25,
        "motion_weight": 0.25
    }

def compute_object_score_yolo(frame):
    results = yolo_model.predict(source=frame, save=False, conf=0.3, verbose=False)
    object_count = len(results[0].boxes) if results else 0
    return object_count

def compute_saliency_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Laplacian(gray, cv2.CV_64F)
    abs_edges = np.abs(edges)
    score = np.mean(abs_edges) / 255.0
    return float(min(score, 1.0))

def compute_motion_score(prev_frame, curr_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.resize(prev_gray, (224, 224))
    curr_gray = cv2.resize(curr_gray, (224, 224))
    diff = cv2.absdiff(prev_gray, curr_gray)
    motion_score = np.mean(diff) / 255.0
    return motion_score

def process_video_chunks(video_path, emotion_score, w1, w2, w3, w4, chunk_duration=5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_seconds = total_frames / fps
    num_chunks = int(total_seconds // chunk_duration)

    print(f"Video has {total_seconds:.2f} seconds -> {num_chunks} chunks")

    all_chunk_scores = []

    for chunk_idx in range(num_chunks):
        frame_indices = [int((chunk_idx * chunk_duration + i) * fps) for i in range(chunk_duration)]
        chunk_scores = []
        prev_frame = None

        for fi in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                break

            saliency_score = compute_saliency_score(frame)
            motion_score = compute_motion_score(prev_frame, frame) if prev_frame is not None else 0.0
            object_score = compute_object_score_yolo(frame)

            chunk_scores.append({
                "frame_index": fi,
                "saliency": saliency_score,
                "motion": motion_score,
                "objects": object_score
            })

            prev_frame = frame

        if not chunk_scores:
            continue

        # Average chunk scores
        avg_saliency = np.mean([s["saliency"] for s in chunk_scores])
        avg_motion = np.mean([s["motion"] for s in chunk_scores])
        avg_objects = np.mean([s["objects"] for s in chunk_scores])

        all_chunk_scores.append({
            "chunk_index": chunk_idx,
            "avg_saliency": avg_saliency,
            "avg_motion": avg_motion,
            "avg_objects": avg_objects
        })

    cap.release()

    # Normalize scores
    max_sal = max([c["avg_saliency"] for c in all_chunk_scores]) or 1
    max_mot = max([c["avg_motion"] for c in all_chunk_scores]) or 1
    max_obj = max([c["avg_objects"] for c in all_chunk_scores]) or 1

    print("\n=== Chunk Scores ===")
    for c in all_chunk_scores:
        norm_sal = c["avg_saliency"] / max_sal
        norm_mot = c["avg_motion"] / max_mot
        norm_obj = c["avg_objects"] / max_obj

        attention_score = (
            w1 * emotion_score +
            w2 * norm_sal +
            w3 * norm_obj +
            w4 * norm_mot
        )

        print(f"\nChunk {c['chunk_index']}:")
        print(f"  Avg Saliency: {c['avg_saliency']:.4f} | Normalized: {norm_sal:.4f}")
        print(f"  Avg Motion:   {c['avg_motion']:.4f} | Normalized: {norm_mot:.4f}")
        print(f"  Avg Objects:  {c['avg_objects']:.2f}   | Normalized: {norm_obj:.4f}")
        print(f"  Attention Score: {attention_score:.4f}")

    return all_chunk_scores

# ======= MAIN =======

if __name__ == "__main__":
    # 1. User input
    user_description = input("Enter a short description of the video: ")
    user_emotion = input("What is the intended emotion? (e.g., funny, sad, inspiring): ")

    # 2. Get weights + emotion score from Mistral
    llm_result = get_llm_weights_and_emotion_score(user_description, user_emotion)
    emotion_score = llm_result["emotion_score"]
    w1 = llm_result["emotion_weight"]
    w2 = llm_result["saliency_weight"]
    w3 = llm_result["object_weight"]
    w4 = llm_result["motion_weight"]

    print("\n=== LLM-Based Weights ===")
    print(f"Emotion Score: {emotion_score}")
    print(f"Weights → Emotion: {w1}, Saliency: {w2}, Objects: {w3}, Motion: {w4}")

    # 3. Process video
    video_path = "6178163-hd_1916_1048_30fps.mp4"  # 🔁 Change to your input
    process_video_chunks(video_path, emotion_score, w1, w2, w3, w4)
