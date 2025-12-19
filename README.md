# Molmo2

Molmo2 is a family of open vision-language models developed by the Allen Institute for AI (Ai2) that support image, video, and multi-image understanding and grounding. Molmo2 models are trained on publicly available third-party datasets and Molmo2 data, a collection of highly-curated image-text and video-text pairs. It has state-of-the-art performance among multimodal models with similar size.

## Model Checkpoints

| Model | Base LLM | Vision Backbone | Notes |
|-------|----------|-----------------|-------|
| `allenai/Molmo2-O-7B` | Olmo3-7B-Instruct | SigLIP 2 | Outperforms others on short videos, counting, and captioning |
| `allenai/Molmo2-4B` | Qwen3-4B-Instruct | SigLIP 2 | Compact model with competitive performance |
| `allenai/Molmo2-8B` | Qwen3-8B | SigLIP 2 | Balanced size and performance |
| `allenai/Molmo2-VideoPoint-4B` | Qwen3-4B-Instruct | SigLIP 2 | Finetuned on Molmo2-VideoPoint data only for video pointing and counting |

All models are competitive on long-videos.

## Installation

**Important:** Requires `transformers==4.57.1`

```bash
pip install transformers==4.57.1
```

```bash
pip install fiftyone umap-learn
```

```bash
pip install einops accelerate decord2 molmo_utils
```

## Usage

### Load a Dataset

```python
import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub

dataset = load_from_hub(
    "Voxel51/qualcomm-interactive-video-dataset",
    max_samples=20,
    overwrite=True
)

# REQUIRED: Compute metadata (needed for timestamp-to-frame conversion)
dataset.compute_metadata()
```

### Load the Model

```python
import fiftyone.zoo as foz

model = foz.load_zoo_model("allenai/Molmo2-4B")
```

### Operations

| Operation | Prompt Template | Output Field Type |
|-----------|-----------------|-------------------|
| `pointing` | `"Point to the {prompt}."` | Frame-level `fo.Keypoints` |
| `tracking` | `"Track the {prompt}."` | Frame-level `fo.Keypoints` with `fo.Instance` linking |
| `describe` | Uses prompt directly | Sample-level string |
| `temporal_localization` | Fixed prompt (finds activity events) | Sample-level `fo.TemporalDetections` |
| `comprehensive` | Fixed prompt (full video analysis) | Sample-level mixed (see below) |

### Output Fields by Operation

**pointing / tracking:**
- Frame-level `fo.Keypoints` stored on each frame
- Each keypoint has `label`, `points` (normalized x, y), and `index` (object ID)
- For tracking, keypoints share `fo.Instance` objects to link across frames

**describe:**
- Sample-level string field containing the model's text response

**temporal_localization:**
- Sample-level `fo.TemporalDetections` with `start`/`end` frame numbers and event descriptions

**comprehensive:**
- `summary`: Sample-level string
- `events`: `fo.TemporalDetections` 
- `objects`: `fo.TemporalDetections` (with first/last appearance times)
- `text_content`: `fo.TemporalDetections` (for any text detected in video)
- `scene_info_*`: `fo.Classification` fields (setting, time_of_day, location_type)
- `activities_*`: `fo.Classification` or `fo.Classifications` fields

### Embeddings

```python
model.pooling_strategy = "mean"  # or "max" or "cls"

dataset.compute_embeddings(
    model,
    batch_size=8,
    num_workers=2,
    embeddings_field="molmo_embeddings",
    skip_failures=False
)
```

### Visualize Embeddings

```python
import fiftyone.brain as fob

results = fob.compute_visualization(
    dataset,
    method="umap",  # Also supports "tsne", "pca"
    brain_key="molmo_viz",
    embeddings="molmo_embeddings",
    num_dims=2  # or 3 for 3D
)
```

### Describe

```python
# With a global prompt
model.operation = "describe"
model.prompt = "Provide a short description for what is happening in the video"

dataset.apply_model(
    model,
    "prompted_describe",
    batch_size=16,
    num_workers=4,
    skip_failures=False
)

# With per-sample prompts from a field
model.operation = "describe"

dataset.apply_model(
    model,
    prompt_field="question",
    label_field="answer_pred",
    batch_size=16,
    num_workers=1,
    skip_failures=False
)
```

### Pointing

```python
model.operation = "pointing"
model.prompt = "person's nose"

dataset.apply_model(
    model,
    "point_pred",
    batch_size=16,
    num_workers=1,
    skip_failures=False
)
```

### Tracking

```python
model.operation = "tracking"
model.prompt = "person's hand"

dataset.apply_model(
    model,
    "track_pred",
    batch_size=16,
    num_workers=1,
    skip_failures=False
)
```

### Comprehensive

```python
model.operation = "comprehensive"

dataset.apply_model(
    model,
    "comprehensive",
    batch_size=2,
    num_workers=1,
    skip_failures=False
)
```

### Temporal Localization

```python
model.operation = "temporal_localization"

dataset.apply_model(
    model,
    "temporal_localization",
    batch_size=2,
    num_workers=1,
    skip_failures=False
)
```

### Launch the App

```python
session = fo.launch_app(dataset, auto=False)
```

## Citation

If you use Molmo2 in your research, please cite the [technical report](https://www.datocms-assets.com/64837/1765901660-molmo_v2_2026-techreport-3.pdf):

```bibtex
@techreport{molmo2,
  title={Molmo2: Open Weights and Data for Vision-Language Models with Video Understanding and Grounding},
  author={Clark, Christopher and Zhang, Jieyu and Ma, Zixian and Park, Jae Sung and Salehi, Mohammadreza and Tripathi, Rohun and Lee, Sangho and Ren, Zhongzheng and Kim, Chris Dongjoo and Yang, Yinuo and Shao, Vincent and Yang, Yue and Huang, Weikai and Gao, Ziqi and Anderson, Taira and Zhang, Jianrui and Jain, Jitesh and Stoica, George and Han, Winston and Farhadi, Ali and Krishna, Ranjay},
  institution={Allen Institute for AI},
  year={2025}
}
```
