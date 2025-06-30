<<<<<<< HEAD
# 🎬 Emotion-Aware Video Chunk Scoring

This Python script performs **emotion-guided video analysis** by dividing a video into chunks and scoring each based on **saliency**, **motion**, and **object presence**—all dynamically weighted using LLM-generated values based on a user description and intended emotion.

## ✨ Features

- 🤖 Uses [Mixtral-8x7B-Instruct](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) to generate emotion weights
- 🧠 Calculates chunk-wise attention scores using:
  - Emotion Score
  - Saliency (Laplacian edge detection)
  - Motion (frame differences)
  - Object Detection (YOLOv8)
- 📦 Outputs per-chunk attention scores for further processing or editing decisions

---

## 🗂️ Directory Structure

```
.
├── .env                   # Contains your HuggingFace API key
├── yolov8n.pt             # YOLOv8n weights file
├── video_analysis.py      # Main script
├── 6178163-hd_1916_1048_30fps.mp4  # Sample video (replaceable)
```

---

## 🛠️ Requirements

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Also, ensure you have Python 3.7+ and a video file.

---

## 🔐 API Key Setup

You need a HuggingFace API key with access to the **Mixtral-8x7B-Instruct model**.  
Create a `.env` file in the root directory with:

```
API_KEY=your_huggingface_api_key_here
```

---

## 🧠 How It Works

1. **User Prompt**: Input a brief video description and target emotion.
2. **LLM Analysis**: Mixtral generates a valid JSON containing:
   - `emotion_score` ∈ [0.0, 1.0]
   - 4 weights (emotion, saliency, object, motion) ≈ 1.0
3. **Video Processing**:
   - The video is split into `5s` chunks.
   - Each chunk is analyzed frame-by-frame for:
     - Saliency (via Laplacian edges)
     - Motion (frame diff)
     - Objects (via YOLOv8)
4. **Scoring**: A normalized attention score is calculated for each chunk:
   ```
   attention_score = 
       w1 * emotion_score +
       w2 * normalized_saliency +
       w3 * normalized_object_count +
       w4 * normalized_motion
   ```
5. **Output**: Prints per-chunk scores and attention weights.

---

## 📦 Example Output

```
Enter a short description of the video: A dog running in a park
What is the intended emotion? (e.g., funny, sad, inspiring): inspiring

=== LLM-Based Weights ===
Emotion Score: 0.83
Weights → Emotion: 0.35, Saliency: 0.30, Objects: 0.20, Motion: 0.15

Video has 25.33 seconds -> 5 chunks

Chunk 0:
  Avg Saliency: 0.1523 | Normalized: 0.8231
  Avg Motion:   0.0341 | Normalized: 0.7120
  Avg Objects:  3.0    | Normalized: 0.7500
  Attention Score: 0.7024
...
```

---

## 📁 Configurable Parameters

You can modify the following directly in the script:

- **Video file**: Replace the `video_path` variable
- **Chunk size**: Change `chunk_duration=5` in `process_video_chunks()`
- **YOLO model**: Swap `"yolov8n.pt"` with another YOLOv8 weight

---

## 🧪 Tips & Notes

- The system falls back to equal weights if the API fails.
- Mixtral sometimes returns malformed JSON; basic regex parsing is applied.
- Only a single YOLO prediction model is loaded to improve efficiency.

---

## 🧊 TODOs / Future Work

- Add CLI support with `argparse`
- Use saliency maps (like EigenCAM) instead of Laplacian edges
- Visualize per-chunk attention for video editing

---

## 🧑‍💻 Author
 
Inspired by multi-factor video scoring workflows used in automatic editors and emotion-aware media tools.
=======
# Video_Editor
An AI assisted video editor
>>>>>>> 615182d10f80228241099f57addcede102f2c433
