<h2 align="center"> <a href="https://arxiv.org/pdf/2505.18110">🤖 [NeurIPS 2025] Watch and Listen: Understanding Audio-Visual-Speech Moments with Multimodal LLM</a></h2>
<h5 align="center"> Zinuo Li<sup>1</sup>, Xian Zhang<sup>1</sup>, Yongxin Guo<sup>2</sup>, Mohammed Bennamoun<sup>1</sup>, Farid Boussaid<sup>1</sup>, Girish Dwivedi<sup>1</sup>, Luqi Gong<sup>3</sup>, Qiuhong Ke<sup>4</sup> </h5>
<h5 align="center">  <sup>1</sup>University of Western Australia, <sup>2</sup>Alibaba Group, <sup>3</sup>Zhejiang Laboratory, <sup>4</sup>Monash University </h5>

<h5 align="center">
  
[![hf_data](https://img.shields.io/badge/🤗-Datasets-9C276A.svg)](https://huggingface.co/datasets/zinuoli/TriSense-2M)
[![arxiv](https://img.shields.io/badge/Arxiv-2505.18110-b31b1b.svg?logo=arXiv)](https://arxiv.org/pdf/2505.18110?)

</h5>

![af355688748a212818b4746797e6731](https://github.com/user-attachments/assets/fbc89818-b878-4efe-b72c-959f35db169e)

## 📢 TODO List
This repo is under construction. We will continuously update this repo for better contribution and performance.

We found the previous LLM backbone was out-of-date, so we are training a more powerful model and planning an extension version to journal.

- [x] Release dataset
- [x] Higher quality data
- [ ] Release trainining code
- [ ] Post-training on highher quality data

## 📢 Dataset Usage
TriSense-2M is a large-scale multimodal dataset for training the TriSense model. It consists of raw data after judger evaluation and three stages of processed training data. We have done another round of filtering compared to the paper version.

## Dataset Sources

- **Repository:** https://github.com/zinuoli/TriSense
- **Paper:** https://arxiv.org/pdf/2505.18110?

## Uses

### Direct Use

This dataset is intended for training and fine-tuning multimodal models on audio-video-speedch video temporal understanding tasks.

### Out-of-Scope Use

1. Law enforcement, surveillance, or authoritarian monitoring systems,
2. Any application that could violate privacy or civil liberties,
3. Behavior tracking, identity resolution, or intent inference.

### Social Impact
1. We highlight that downstream applications must be carefully audited for demographic fairness.
2. We caution that TriSense is a research prototype and is not intended for deployment without further fairness evaluations.
3. We encourage the use of context-sensitive moderation policies and fairness-aware benchmarks in real-world settings.

## Dataset Structure

The dataset contains four JSON files:

| File | Description |
|------|-------------|
| `TriSense-2M-After-Judger.json` | Raw data after judger evaluation |
| `stage1.json` | Stage 1 — Multimodal Alignment |
| `stage2.json` | Stage 2 — Training Quey-Based Connector |
| `stage3.json` | Stage 3 — Traning LLM backbone |

### Data Flow

```
TriSense-2M-After-Judger.json  
        │
        ├──► stage1.json      
        ├──► stage2.json       
        └──► stage3.json       
```

---

## Data Fields

### TriSense-2M-After-Judger.json

Raw data produced after the judger evaluates model outputs. Each record corresponds to a video segment with multi-modal annotations and judger evaluation results.

| Field | Type | Description |
|-------|------|-------------|
| `video` | string | Video identifier (each str represents a YouTube ID) |
| `times` | array | List of `[start, end]` temporal segments in seconds |
| `audio` | string | Audio modality description |
| `visual` | string | Visual modality description |
| `speech` | string | Speech/transcript content |
| `original_avs` | string | Ground-truth AVS (Audio-Visual-Speech) caption |
| `original_av` | string | Ground-truth AV (Audio-Visual) caption |
| `original_vs` | string | Ground-truth VS (Visual-Speech) caption |
| `model_response` | string | Raw model output (JSON string) |
| `eval_result` | object | Judger evaluation for AVS, AV, VS modalities. Each contains `caption`, `score`, `decision` (KEEP/REJECT). Some records may contain `raw_response` when parsing fails. |
| `success` | boolean | Whether judger evaluation succeeded |

**Example structure:**

```json
{
  "video": "mPcah3P2D-E",
  "times": [[38.0, 64.1]],
  "audio": "Sounds like a man is speaking...",
  "visual": "a man is standing in front of a bus...",
  "speech": "that use Shimano like Byron Merida...",
  "original_avs": "Shimano helps its teams with time trial technology...",
  "original_av": "A man is speaking and breathing...",
  "original_vs": "A man is discussing bike components...",
  "model_response": "{...}",
  "eval_result": {
    "AVS": {"caption": "...", "score": 5, "decision": "KEEP"},
    "AV": {"caption": "...", "score": 4, "decision": "KEEP"},
    "VS": {"caption": "...", "score": 5, "decision": "KEEP"}
  },
  "success": true
}
```

---

### stage1.json

Stage 1 data for image captioning. Uses the `<video>` placeholder for visual input (images treated as single-frame videos).

| Field | Type | Description |
|-------|------|-------------|
| `image` | string | Relative path to the image file (e.g., `images/00004/000048746.jpg`) |
| `conversations` | array | Alternating human/gpt turns; human uses `<video>\n{instruction}` |
| `times` | array | Empty `[[]]` for static images (no temporal info) |

**Example structure:**

```json
{
  "image": "images/00004/000048746.jpg",
  "conversations": [
    {"from": "human", "value": "<video>\nPresent a compact description of the photo's key features."},
    {"from": "gpt", "value": "<sync><time>the north face men's ultra trail running shoes, black"}
  ],
  "times": [[]]
}
```

---

### stage2.json

Stage 2 data for video temporal understanding. Multi-turn conversations with explicit temporal segments.

| Field | Type | Description |
|-------|------|-------------|
| `video` | string | Video file path (e.g., `IwUIKDTErNo.mp4`) |
| `conversations` | array | Multi-turn QA; human prompts include time ranges like `[795.6, 802.1]` |
| `times` | array | One `[start, end]` pair per turn, aligned with conversations |

**Task types:** AVS summary (visual + audio + speech), VS summary (visual + speech), AV summary (visual + audio), visual-only event description, temporal grounding (find timestamp for given caption).

**Response format:** GPT responses use `<sync><time><time>...` tokens for temporal alignment with the video.

**Example structure:**

```json
{
  "video": "IwUIKDTErNo.mp4",
  "conversations": [
    {"from": "human", "value": "<video>\nBetween [795.6, 802.1], I need a summary of the video..."},
    {"from": "gpt", "value": "<sync><time><time>...Miyoko Miyazawa, a nurse for almost 40 years..."}
  ],
  "times": [[795.6, 802.1], [795.6, 802.1]]
}
```

---

### stage3.json

Stage 3 data for advanced multi-modal reasoning and grounding. Supports both video-level QA and temporal segment tasks.

| Field | Type | Description |
|-------|------|-------------|
| `video` | string | Video file path |
| `conversations` | array | Multi-turn conversations |
| `times` | array | `[]` for video-level questions; `[start, end]` for temporal segment tasks |
| `id` | int | Optional sample identifier |

**Task types:** Video-level QA (no temporal segment; `times` entry is `[]`), temporal captioning (summarize segment with AVS/VS/AV), temporal grounding (locate timestamp for given caption), multi-modal reasoning (visual, audio, speech).

**Response format:** Same as stage2 — `<sync><time><time>...` for temporal alignment.

**Example structure:**

```json
{
  "video": "activitynet/v_7LmSZAoD6-c.mp4",
  "conversations": [
    {"from": "human", "value": "<video>\nDoes the athlete run quickly or slowly during her run-up for the high jump?"},
    {"from": "gpt", "value": "<sync><time>The athlete runs quickly during her run-up for the high jump."}
  ],
  "times": [[], []],
  "id": 0
}
```
