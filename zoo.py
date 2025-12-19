import logging
import re
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import numpy as np

import fiftyone as fo
import fiftyone.core.labels as fol
import fiftyone.core.models as fom
import fiftyone.utils.torch as fout
from fiftyone.core.models import SupportsGetItem, TorchModelMixin, SamplesMixin
from fiftyone.utils.torch import GetItem

from transformers import AutoProcessor, AutoModelForImageTextToText
from molmo_utils import process_vision_info

logger = logging.getLogger(__name__)


def get_device():
    """Get the best available device for inference.
    
    Returns:
        str: Device name ("cuda", "mps", or "cpu")
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# =============================================================================
# Regex patterns for parsing Molmo2 output
# =============================================================================

COORD_REGEX = re.compile(r"<(?:points|tracks).*? coords=\"([0-9\t:;, .]+)\"/?>")
FRAME_REGEX = re.compile(r"(?:^|\t|:|,|;)([0-9\.]+) ([0-9\. ]+)")
POINTS_REGEX = re.compile(r"([0-9]+) ([0-9]{3,4}) ([0-9]{3,4})")


def extract_video_points(text: str, image_w: int, image_h: int, extract_ids: bool = True) -> List[Tuple]:
    """Extract video pointing coordinates from model output text.
    
    Args:
        text: Raw model output text containing <points> or <tracks> tags
        image_w: Video frame width for coordinate conversion
        image_h: Video frame height for coordinate conversion
        extract_ids: If True, return (timestamp, idx, x, y); else (timestamp, x, y)
    
    Returns:
        List of tuples: (timestamp, idx, x, y) or (timestamp, x, y)
        - timestamp: Time in seconds (float)
        - idx: Object ID (int)
        - x, y: Pixel coordinates (float)
    """
    all_points = []
    for coord in COORD_REGEX.finditer(text):
        for point_grp in FRAME_REGEX.finditer(coord.group(1)):
            frame_id = float(point_grp.group(1))  # Timestamp in seconds
            for match in POINTS_REGEX.finditer(point_grp.group(2)):
                idx = int(match.group(1))
                # Convert from 0-1000 scale to pixel coordinates
                x = float(match.group(2)) / 1000 * image_w
                y = float(match.group(3)) / 1000 * image_h
                if 0 <= x <= image_w and 0 <= y <= image_h:
                    if extract_ids:
                        all_points.append((frame_id, idx, x, y))
                    else:
                        all_points.append((frame_id, x, y))
    return all_points


def has_grounded_output(text: str) -> bool:
    """Check if model output contains <points> or <tracks> tags."""
    return bool(COORD_REGEX.search(text))

def normalize_timestamp(timestamp_str: str) -> float:
    """Convert timestamp to seconds."""
    s = str(timestamp_str).strip()
    
    if ':' in s:
        parts = s.split(':')
        return sum(float(p) * m for p, m in zip(reversed(parts), [1, 60, 3600]))
    
    # Extract first valid number (handles malformed "2.00.00" → 2.00)
    match = re.match(r'(\d+\.?\d*)', s)
    return float(match.group(1)) if match else 0.0


def extract_json(text: str):
    """Extract JSON from model output.
    
    Handles multiple formats:
    1. JSON wrapped in markdown code blocks: ```json {...} ```
    2. JSON object starting with { and ending with }
    3. JSON array starting with [ and ending with ]
    4. Raw JSON text
    
    Args:
        text: Model output text
    
    Returns:
        Parsed JSON object (dict or list), or None if parsing fails
    """
    import json
    
    # Try to find JSON in markdown code block first
    json_match = re.search(r'```json\s*(\{.*?\}|\[.*?\])\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try to extract JSON object or array from text
    # Look for outermost { } or [ ]
    for pattern in [
        r'\{.*\}',  # Object
        r'\[.*\]',  # Array
    ]:
        json_match = re.search(pattern, text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                continue
    
    # Try parsing the entire text as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None




# =============================================================================
# Prompt templates
# =============================================================================

OPERATION_PROMPTS = {
    "pointing": "Point to the {prompt}.",
    "tracking": "Track the {prompt}.",
    "describe": None,  # Use prompt directly, no template
    "temporal_localization": """Localize activity events in the video. Output start and end timestamp for each event.
Output ONLY valid JSON in this format (no additional text):
[{"start": "mm:ss.ff", "end": "mm:ss.ff", "description": "event description"}]""",
    "comprehensive": """Analyze this video comprehensively. Output ONLY valid JSON in this exact format:

{
  "summary": "Brief description of the video",
  "objects": [{"name": "object name", "first_appears": "mm:ss.ff", "last_appears": "mm:ss.ff"}],
  "events": [{"start": "mm:ss.ff", "end": "mm:ss.ff", "description": "event description"}],
  "text_content": [{"start": "mm:ss.ff", "end": "mm:ss.ff", "text": "text content"}],
  "scene_info": {"setting": "<one-word-description>", "time_of_day": "<one-word-description>", "location_type": "<one-word-description>"},
  "activities": {"primary_activity": "activity name", "secondary_activities": "comma-separated activities"}
}

Do not include any text before or after the JSON.""",
}


# =============================================================================
# GetItem class for DataLoader
# =============================================================================

class Molmo2GetItem(GetItem):
    """GetItem transform for Molmo2 video model.
    
    Returns video filepath and optional prompt for processing in predict_all().
    Heavy video processing happens on GPU, not in DataLoader workers.
    
    The required_keys are STATIC - they don't change at runtime.
    FiftyOne uses field_mapping to map these static keys to actual dataset fields.
    """
    
    @property
    def required_keys(self):
        """Fields needed from each sample (STATIC).
        
        These keys are logical names that FiftyOne maps to actual dataset fields
        via field_mapping. For example:
        - field_mapping = {"prompt_field": "my_questions"}
        - sample_dict["prompt_field"] will contain sample.my_questions
        """
        return ["filepath", "prompt_field", "metadata"]
    
    def __call__(self, sample_dict):
        """Return video filepath, optional prompt, and metadata for processing.
        
        Args:
            sample_dict: Dict with keys mapped by FiftyOne via field_mapping
                        - "filepath" always present
                        - "prompt_field" present if user passed prompt_field to apply_model
                        - "metadata" video metadata (frame_rate, etc.)
        
        Returns:
            dict: {"filepath": str, "prompt": str or None, "metadata": VideoMetadata or None}
        """
        return {
            "filepath": sample_dict["filepath"],
            "prompt": sample_dict.get("prompt_field"),  # None if not in mapping
            "metadata": sample_dict.get("metadata"),  # None if not computed
        }


# =============================================================================
# Model Configuration
# =============================================================================

class Molmo2VideoModelConfig(fout.TorchImageModelConfig):
    """Configuration for Molmo2 video model.
    
    Operations:
        - "pointing": Point to objects → frame-level fo.Keypoints
        - "tracking": Track objects → frame-level fo.Keypoints with fo.Instance
        - "describe": Free-form text output → sample-level string field
        - "temporal_localization": Find events with timestamps → sample-level fo.TemporalDetections
        - "comprehensive": Mixed analysis → sample-level text + fo.TemporalDetections
    
    Key Parameters:
        model_path: HuggingFace model ID (default: "allenai/Molmo2-4B")
        operation: One of the operations above (default: "pointing")
        prompt: Text prompt - used with templates for pointing/tracking,
                directly for other operations
        pooling_strategy: Strategy for pooling embeddings - "cls", "mean", or "max"
                         (default: "mean")
    
    Note:
        Parameters can be set after instantiation via model properties.
    """
    
    def __init__(self, d):
        if "raw_inputs" not in d:
            d["raw_inputs"] = True
        super().__init__(d)
        
        # Model path
        self.model_path = self.parse_string(d, "model_path", default="allenai/Molmo2-4B")
        
        # Operation: "pointing", "tracking", or "describe"
        self.operation = self.parse_string(d, "operation", default="pointing")
        
        # Prompt for all operations (can be set later)
        self.prompt = self.parse_string(d, "prompt", default=None)
        
        # Pooling strategy for embeddings
        self.pooling_strategy = self.parse_string(d, "pooling_strategy", default="mean")
        
        # Validate pooling strategy
        if self.pooling_strategy not in ["cls", "mean", "max"]:
            raise ValueError(
                f"pooling_strategy must be 'cls', 'mean', or 'max', "
                f"got '{self.pooling_strategy}'"
            )
        

# =============================================================================
# Main Model Class
# =============================================================================

class Molmo2VideoModel(fom.Model, SamplesMixin, SupportsGetItem, TorchModelMixin):
    """FiftyOne wrapper for Molmo2 video understanding model.
    
    Operations and Outputs:
        - pointing: fo.Keypoints per frame (points at objects)
        - tracking: fo.Keypoints per frame with fo.Instance linking
        - describe: Plain text string (sample-level)
        - temporal_localization: fo.TemporalDetections (events with time intervals)
        - comprehensive: Mixed output (text summary + temporal events)
    
    Prompt Handling:
        All operations use a single `prompt` parameter. For pointing/tracking,
        the prompt is inserted into a template ("Point to the {prompt}." or
        "Track the {prompt}."). For describe, the prompt is used directly.
        
        Prompts can be set globally or per-sample:
        - Global: model.prompt = "red cars"
        - Per-sample: pass prompt_field to apply_model()
        
        Per-sample prompts take priority over global.
    
    Examples:
        # Global prompt with pointing
        model.operation = "pointing"
        model.prompt = "red cars"  # → "Point to the red cars."
        dataset.apply_model(model, label_field="points")
        
        # Per-sample prompts with tracking
        model.operation = "tracking"
        dataset.apply_model(model, label_field="tracks", prompt_field="objects")
        # sample.objects = "person" → "Track the person."
        
        # Describe (no template)
        model.operation = "describe"
        model.prompt = "What is happening?"
        dataset.apply_model(model, label_field="descriptions")
    
    Batching Architecture:
        - GetItem returns filepath + optional prompt (lightweight)
        - collate_fn returns list as-is (no stacking)
        - predict_all processes videos on GPU
    """
    
    def __init__(self, config):
        # Initialize mixins
        SamplesMixin.__init__(self)
        SupportsGetItem.__init__(self)
        
        # REQUIRED: Preprocessing flag
        self._preprocess = False
        
        # Store configuration
        self.config = config
        
        # Store global default prompt (for fallback)
        self.default_prompt = config.prompt
        
        # Detect best available device
        self.device = get_device()
        logger.info(f"Using device: {self.device}")
        
        # Lazy-loaded model components
        self._processor = None
        self._model = None
        
        # Embeddings cache
        self._last_computed_embeddings = None
        
        # Fields needed from samples (for SamplesMixin if we add it back)
        self._fields = {}
    
    @property
    def needs_fields(self):
        """A dict mapping model-specific keys to sample field names."""
        return self._fields
    
    @needs_fields.setter
    def needs_fields(self, fields):
        self._fields = fields

    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, *args):
        """Context manager exit - clear GPU memory cache."""
        # Clear cache based on device type (don't move model to CPU)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return False
    
    # =========================================================================
    # Properties from Model base class
    # =========================================================================
    
    @property
    def media_type(self):
        """Media type this model operates on."""
        return "video"
    
    @property
    def transforms(self):
        """Preprocessing transforms (None for SupportsGetItem)."""
        return None
    
    @property
    def preprocess(self):
        """Whether model applies preprocessing."""
        return self._preprocess
    
    @preprocess.setter
    def preprocess(self, value):
        """Allow FiftyOne to control preprocessing."""
        self._preprocess = value
    
    @property
    def ragged_batches(self):
        """MUST be False to enable batching!
        
        Even though videos have variable lengths, we return False.
        Variable sizes are handled via custom collate_fn.
        """
        return False
    
    # =========================================================================
    # Operation configuration properties (can be set after instantiation)
    # =========================================================================
    
    @property
    def operation(self):
        """Current operation type: 'pointing', 'tracking', or 'describe'."""
        return self.config.operation
    
    @operation.setter
    def operation(self, value):
        """Set operation type."""
        valid_ops = ["pointing", "tracking", "describe", "temporal_localization", "comprehensive"]
        if value not in valid_ops:
            raise ValueError(f"operation must be one of {valid_ops}, got '{value}'")
        self.config.operation = value
    
    @property
    def prompt(self):
        """Global prompt for all operations.
        
        For pointing/tracking, this is inserted into the operation template.
        For describe, this is used directly.
        """
        return self.default_prompt
    
    @prompt.setter
    def prompt(self, value):
        """Set global prompt."""
        self.default_prompt = value
    
    @property
    def pooling_strategy(self):
        """Get pooling strategy for embeddings."""
        return self.config.pooling_strategy
    
    @pooling_strategy.setter
    def pooling_strategy(self, value):
        """Set pooling strategy for embeddings.
        
        Args:
            value: Pooling strategy ("cls", "mean", or "max")
        """
        if value not in ["cls", "mean", "max"]:
            raise ValueError(
                f"pooling_strategy must be 'cls', 'mean', or 'max', got '{value}'"
            )
        self.config.pooling_strategy = value
    
    # =========================================================================
    # Properties from TorchModelMixin (Critical for variable sizes!)
    # =========================================================================
    
    @property
    def has_collate_fn(self):
        """Tell FiftyOne we have custom collation."""
        return True
    
    @property
    def collate_fn(self):
        """Return batch as-is (don't stack).
        
        This prevents FiftyOne from trying np.stack() on video paths.
        """
        @staticmethod
        def identity_collate(batch):
            return batch
        return identity_collate
    
    # =========================================================================
    # Methods from SupportsGetItem
    # =========================================================================
    
    def build_get_item(self, field_mapping=None):
        """Build GetItem transform for data loading.
        
        Args:
            field_mapping: Dict mapping static keys to dataset field names.
                          Handled entirely by FiftyOne and parent GetItem class.
        
        Returns:
            Molmo2GetItem instance
        """
        return Molmo2GetItem(field_mapping=field_mapping)
    
    # =========================================================================
    # Model loading
    # =========================================================================
    
    def _load_model(self):
        """Load Molmo2 model and processor from HuggingFace."""
        logger.info(f"Loading Molmo2 model from {self.config.model_path} on {self.device}")
        
        # Load processor
        self._processor = AutoProcessor.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
            dtype="auto",
            device_map=self.device
        )
        
        # Load model on detected device
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
            dtype="auto",
        ).to(self.device).eval()
        
        logger.info(f"Model loaded on {self.device}")
    
    # =========================================================================
    # Helper methods
    # =========================================================================
    
    def _extract_video_path(self, video):
        """Extract filepath from video reader object or string.
        
        When predict() is called by FiftyOne's apply_model(), it receives a video
        reader object (e.g., FFmpegVideoReader) instead of a string path. Video
        readers store the actual file path in their 'inpath' attribute.
        
        Note: This is different from sample.filepath - that's on the Sample object,
        not the video reader object.
        
        Args:
            video: Video reader object (has 'inpath' attribute) or string path
            
        Returns:
            str: Video file path on disk
        """
        if isinstance(video, str):
            return video
        elif hasattr(video, 'inpath'):
            # Standard attribute for video readers (FFmpegVideoReader, etc.)
            return video.inpath
        else:
            raise TypeError(
                f"Cannot extract filepath from {type(video).__name__}. "
                f"Expected string path or video reader with 'inpath' attribute."
            )
    
    # =========================================================================
    # Embeddings support
    # =========================================================================
    
    @property
    def has_embeddings(self):
        """Whether this instance can generate embeddings."""
        return True
    
    def _apply_pooling(self, hidden_states):
        """Apply pooling strategy to hidden states.
        
        Converts variable-length hidden states to fixed-dimension embeddings.
        
        Args:
            hidden_states: Tensor of shape (batch, seq_len, hidden_dim)
        
        Returns:
            torch.Tensor: Pooled embeddings of shape (batch, hidden_dim)
        """
        strategy = self.pooling_strategy
        
        if strategy == "cls":
            # Use first token (CLS token)
            pooled = hidden_states[:, 0, :]
        elif strategy == "mean":
            # Average pooling across sequence
            pooled = hidden_states.mean(dim=1)
        elif strategy == "max":
            # Max pooling across sequence
            pooled = hidden_states.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling strategy: {strategy}")
        
        # Normalize for cosine similarity
        pooled = F.normalize(pooled, p=2, dim=1)
        
        return pooled
    
    def embed_video(self, video_path, prompt=None):
        """Embed a single video.
        
        Args:
            video_path: Path to video file (str)
            prompt: Optional text prompt to condition the embedding
                   If None, uses default description prompt
        
        Returns:
            numpy array: 1D embedding vector with shape (hidden_dim,)
        """
        # Lazy load model
        if self._model is None:
            self._load_model()
        
        # Use default prompt if none provided
        if prompt is None:
            prompt = "Provide a description of what is happening in this video"
        
        # Build messages
        messages = self._build_messages(video_path, prompt)
        
        # Process video
        _, videos, video_kwargs = process_vision_info(messages)
        videos, video_metadatas = zip(*videos)
        videos, video_metadatas = list(videos), list(video_metadatas)
        
        # Apply chat template
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process inputs
        inputs = self._processor(
            videos=videos,
            video_metadata=video_metadatas,
            text=text,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass through encoder (no generation)
        with torch.no_grad():
            # Get model outputs with hidden states
            outputs = self._model.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            
            # Get hidden states from last layer
            hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)
            
            # Apply pooling to get fixed-dimension embedding
            embedding = self._apply_pooling(hidden_states)
            
            # Convert to float32 (numpy doesn't support bfloat16), then to numpy
            return embedding.cpu().float().numpy()[0]
    
    def embed_videos(self, video_paths, prompt=None):
        """Embed multiple videos.
        
        Args:
            video_paths: List of video file paths
            prompt: Optional text prompt (same for all videos)
        
        Returns:
            numpy array: 2D array of embeddings with shape (num_videos, hidden_dim)
        """
        # Process videos sequentially for reliability
        embeddings = []
        
        for video_path in video_paths:
            embedding = self.embed_video(video_path, prompt=prompt)
            embeddings.append(embedding)
        
        # Stack into 2D array
        result = np.stack(embeddings, axis=0)
        
        # Cache for get_embeddings()
        self._last_computed_embeddings = result
        
        return result
    
    def embed(self, video, prompt=None):
        """Embed a single video.
        
        FiftyOne calls this method for single-sample embedding.
        
        Args:
            video: Video reader object or string path
            prompt: Optional text prompt
        
        Returns:
            numpy array: 1D embedding vector
        """
        video_path = self._extract_video_path(video)
        return self.embed_video(video_path, prompt=prompt)
    
    def embed_all(self, videos, prompt=None):
        """Embed multiple videos.
        
        FiftyOne calls this method for batch embedding.
        
        Args:
            videos: List of video reader objects or string paths
            prompt: Optional text prompt
        
        Returns:
            numpy array: 2D embeddings with shape (num_videos, hidden_dim)
        """
        video_paths = [self._extract_video_path(video) for video in videos]
        return self.embed_videos(video_paths, prompt=prompt)
    
    def get_embeddings(self):
        """Get the last computed embeddings.
        
        Returns:
            numpy array: The last computed embeddings
        
        Raises:
            ValueError: If no embeddings have been computed yet
        """
        if not self.has_embeddings:
            raise ValueError("This model instance does not expose embeddings")
        
        if self._last_computed_embeddings is None:
            raise ValueError(
                "No embeddings have been computed yet. "
                "Call embed() or embed_all() first."
            )
        
        return self._last_computed_embeddings
    
    # =========================================================================
    # Prompt building
    # =========================================================================
    
    def _get_prompt(self, prompt=None) -> str:
        """Get final prompt for inference.
        
        For operations with fixed templates (temporal_localization, comprehensive):
        - Use the template directly, no user prompt needed
        
        For operations requiring user input (pointing, tracking, describe):
        - Priority: per-sample > global > error
        - pointing/tracking: format with template
        - describe: use directly
        
        Args:
            prompt: Optional per-sample prompt from batch item
        
        Returns:
            str: Final prompt ready for model inference
        
        Raises:
            ValueError: If no prompt available for operations requiring it
        """
        # Get operation template
        template = OPERATION_PROMPTS.get(self.config.operation)
        
        # Operations with fixed templates (no user prompt needed)
        if self.config.operation in ["temporal_localization", "comprehensive"]:
            return template
        
        # Operations requiring user prompt
        # Step 1: Resolve prompt value
        resolved = prompt if prompt is not None else self.default_prompt
        
        if resolved is None:
            raise ValueError(
                "No prompt provided. Set model.prompt or pass prompt_field to apply_model()."
            )
        
        # Step 2: Apply template if present
        if template is not None:
            return template.format(prompt=resolved)
        
        # No template (describe operation)
        return str(resolved)
    
    def _build_messages(self, video_path: str, prompt: str) -> List[Dict]:
        """Build message structure for Molmo2 inference.
        
        Args:
            video_path: Path to video file
            prompt: Text prompt for generation
        
        Returns:
            List of message dicts in Molmo2 format
        """
        return [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "video", "video": video_path},
            ]
        }]
    
    # =========================================================================
    # Inference
    # =========================================================================
    
    def _run_inference(self, video_path: str, prompt: str) -> Tuple[str, Dict]:
        """Run model inference on a single video.
        
        Args:
            video_path: Path to video file
            prompt: Text prompt for generation
        
        Returns:
            Tuple of (generated_text, video_metadata)
        """
        messages = self._build_messages(video_path, prompt)
        
        # Process video using molmo_utils
        _, videos, video_kwargs = process_vision_info(messages)
        videos, video_metadatas = zip(*videos)
        videos, video_metadatas = list(videos), list(video_metadatas)
        
        # Apply chat template
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process inputs
        inputs = self._processor(
            videos=videos,
            video_metadata=video_metadatas,
            text=text,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=32768  # Allow enough tokens for coordinate generation
            )
        
        # Decode
        generated_tokens = generated_ids[0, inputs['input_ids'].size(1):]
        generated_text = self._processor.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )
        
        return generated_text, video_metadatas[0]
    
    # =========================================================================
    # Output parsing
    # =========================================================================
    
    def _parse_temporal_detections(self, items, metadata, label_type="events") -> fol.TemporalDetections:
        """Parse temporal detections from JSON output.
        
        Args:
            items: List of dicts with temporal information
            metadata: VideoMetadata object (for timestamp to frame conversion)
            label_type: Type of temporal data ("events", "objects", or "text")
        
        Returns:
            fo.TemporalDetections or None
        """
        detections = []
        
        for item in items:
            # Determine timestamps and label based on type
            if label_type == "events":
                start = item.get("start", "00:00.00")
                end = item.get("end", "00:00.00")
                label = str(item.get("description", "event"))
            elif label_type == "objects":
                start = item.get("first_appears", "00:00.00")
                end = item.get("last_appears", "00:00.00")
                label = str(item.get("name", "object"))
            else:  # text
                start = item.get("start", "00:00.00")
                end = item.get("end", "00:00.00")
                label = str(item.get("text", "text"))
            
            # Normalize timestamps (add .00 if missing fractional seconds)
            start = normalize_timestamp(start)
            end = normalize_timestamp(end)
            
            # Use FiftyOne's from_timestamps() which handles both
            # "HH:MM:SS.XXX" strings and numeric seconds
            # It automatically converts to frame numbers using metadata
            detection = fol.TemporalDetection.from_timestamps(
                [start, end], label=label, metadata=metadata
            )
            detections.append(detection)
        
        return fol.TemporalDetections(detections=detections) if detections else None
    
    def _parse_pointing_output(
        self, 
        text: str, 
        video_metadata: Dict, 
        video_fps: float,
        label: str = "object"
    ) -> Dict:
        """Parse <points> output to frame-level fo.Keypoints.
        
        Args:
            text: Model output text
            video_metadata: Dict with "width" and "height"
            video_fps: Video frame rate for timestamp conversion
            label: Label for the keypoints (from the prompt)
        
        Returns:
            Dict with frame numbers as keys, {"points": fo.Keypoints} as values
        """
        image_w = video_metadata["width"]
        image_h = video_metadata["height"]
        
        points = extract_video_points(text, image_w, image_h, extract_ids=True)
        
        if not points:
            return {"points": fol.Keypoints(keypoints=[])}
        
        # Group by frame number
        frame_keypoints = {}
        max_idx = 0
        
        for timestamp, idx, x, y in points:
            max_idx = max(max_idx, idx)
            
            # Convert timestamp (seconds) to 1-based frame number
            frame_number = int(timestamp * video_fps) + 1
            
            # Normalize coordinates to 0-1 scale
            x_norm = x / image_w
            y_norm = y / image_h
            
            if frame_number not in frame_keypoints:
                frame_keypoints[frame_number] = []
            
            frame_keypoints[frame_number].append({
                "idx": idx,
                "x": x_norm,
                "y": y_norm,
            })
        
        # Build FiftyOne labels
        result = {}
        for frame_num, kps in frame_keypoints.items():
            keypoints_list = []
            for kp in kps:
                keypoint = fol.Keypoint(
                    label=label,
                    points=[(kp["x"], kp["y"])],
                    index=kp["idx"],
                )
                keypoints_list.append(keypoint)
            
            result[frame_num] = {"points": fol.Keypoints(keypoints=keypoints_list)}
        
        # Note: Count (max_idx + 1) can be computed from the keypoints by users
        # The index attribute on each keypoint contains the object ID
        
        return result
    
    def _parse_tracking_output(
        self, 
        text: str, 
        video_metadata: Dict, 
        video_fps: float,
        label: str = "tracked"
    ) -> Dict:
        """Parse <tracks> output to frame-level fo.Keypoints with fo.Instance linking.
        
        Args:
            text: Model output text
            video_metadata: Dict with "width" and "height"
            video_fps: Video frame rate for timestamp conversion
            label: Label for the keypoints (from the prompt)
        
        Returns:
            Dict with frame numbers as keys, {"tracks": fo.Keypoints} as values
        """
        image_w = video_metadata["width"]
        image_h = video_metadata["height"]
        
        points = extract_video_points(text, image_w, image_h, extract_ids=True)
        
        if not points:
            return {"tracks": fol.Keypoints(keypoints=[])}
        
        # Create one fo.Instance per unique track ID
        track_instances = {}
        for timestamp, idx, x, y in points:
            if idx not in track_instances:
                track_instances[idx] = fol.Instance()
        
        # Group by frame number with instance linking
        frame_labels = {}
        
        for timestamp, idx, x, y in points:
            frame_num = int(timestamp * video_fps) + 1
            
            # Normalize coordinates
            x_norm = x / image_w
            y_norm = y / image_h
            
            keypoint = fol.Keypoint(
                label=label,
                points=[(x_norm, y_norm)],
                instance=track_instances[idx],  # Link to track instance
                index=idx,
            )
            
            if frame_num not in frame_labels:
                frame_labels[frame_num] = {"tracks": fol.Keypoints(keypoints=[])}
            
            frame_labels[frame_num]["tracks"].keypoints.append(keypoint)
        
        return frame_labels
    
    def _parse_output(self, text: str, video_metadata: Dict, metadata, label: str = "object") -> Dict:
        """Parse model output based on operation type.
        
        Args:
            text: Model output text
            video_metadata: Dict with "width" and "height"
            metadata: VideoMetadata object (for FPS and temporal conversion)
            label: Label for keypoints (from the resolved prompt)
        
        Returns:
            Dict with sample-level (string keys) and frame-level (int keys) labels
        """
        # Get video FPS from metadata
        video_fps = 30.0  # Default
        if metadata and hasattr(metadata, 'frame_rate') and metadata.frame_rate:
            video_fps = metadata.frame_rate
        
        if self.config.operation == "describe":
            # Plain text output
            return {"description": text}
        
        elif self.config.operation == "pointing":
            # Check if output has grounded coordinates
            if has_grounded_output(text):
                return self._parse_pointing_output(text, video_metadata, video_fps, label=label)
            else:
                # Model returned text instead of coordinates
                return {"description": text}
        
        elif self.config.operation == "tracking":
            if has_grounded_output(text):
                return self._parse_tracking_output(text, video_metadata, video_fps, label=label)
            else:
                return {"description": text}
        
        elif self.config.operation == "temporal_localization":
            # Parse JSON output to temporal detections
            json_data = extract_json(text)
            if json_data:
                # Handle both list and dict with "events" key
                items = json_data if isinstance(json_data, list) else json_data.get("events", [])
                if items:
                    temporal_dets = self._parse_temporal_detections(items, metadata)
                    if temporal_dets:
                        return {"events": temporal_dets}
            # No valid JSON or no events - return empty detections
            return {"events": fol.TemporalDetections(detections=[])}
        
        elif self.config.operation == "comprehensive":
            # Parse JSON output with flexible schema
            json_data = extract_json(text)
            if json_data and isinstance(json_data, dict):
                result = {}
                
                # Parse summary (plain text)
                if "summary" in json_data:
                    result["summary"] = str(json_data["summary"])
                
                # Parse events (temporal detections)
                if "events" in json_data and json_data["events"]:
                    temporal_dets = self._parse_temporal_detections(json_data["events"], metadata, label_type="events")
                    if temporal_dets:
                        result["events"] = temporal_dets
                
                # Parse objects (temporal detections with first_appears/last_appears)
                if "objects" in json_data and json_data["objects"]:
                    temporal_dets = self._parse_temporal_detections(json_data["objects"], metadata, label_type="objects")
                    if temporal_dets:
                        result["objects"] = temporal_dets
                
                # Parse text_content (temporal detections)
                if "text_content" in json_data and json_data["text_content"]:
                    temporal_dets = self._parse_temporal_detections(json_data["text_content"], metadata, label_type="text")
                    if temporal_dets:
                        result["text_content"] = temporal_dets
                
                # Parse scene_info (nested dict → Classifications)
                if "scene_info" in json_data and isinstance(json_data["scene_info"], dict):
                    for key, value in json_data["scene_info"].items():
                        field_name = f"scene_info_{key}"
                        result[field_name] = fol.Classification(label=str(value))
                
                # Parse activities (nested dict → Classifications)
                if "activities" in json_data and isinstance(json_data["activities"], dict):
                    for key, value in json_data["activities"].items():
                        field_name = f"activities_{key}"
                        if key.endswith("activities") and isinstance(value, str):
                            # Parse comma-separated values
                            items = [item.strip() for item in value.split(',') if item.strip()]
                            result[field_name] = fol.Classifications(
                                classifications=[fol.Classification(label=item) for item in items]
                            )
                        else:
                            result[field_name] = fol.Classification(label=str(value))
                
                # Return parsed results or use raw text as summary if nothing parsed
                if result:
                    return result
                else:
                    logger.warning("Comprehensive: Empty JSON object, using raw text as summary")
                    return {"summary": text}
            
            # No valid JSON - log warning and use raw text as summary
            logger.warning("Comprehensive: Could not parse JSON, using raw text as summary")
            return {"summary": text}
        
        return {"description": text}
    
    # =========================================================================
    # Main inference methods
    # =========================================================================
    
    def predict(self, arg, sample=None):
        """Single video inference.
        
        Args:
            arg: Video filepath, video reader object, or dict from GetItem
            sample: FiftyOne sample (for metadata)
        
        Returns:
            Dict with labels
        """
        # Convert to batch item format
        if isinstance(arg, dict):
            # Already in correct format from GetItem (batched mode)
            batch_item = arg
        else:
            # Non-batched mode: extract filepath, prompt, and metadata from sample
            if isinstance(arg, str):
                filepath = arg
            else:
                # Video reader object
                filepath = self._extract_video_path(arg)
            
            # Extract prompt from sample if available (via needs_fields)
            prompt = None
            if sample is not None and "prompt_field" in self._fields:
                field_name = self._fields["prompt_field"]
                if sample.has_field(field_name):
                    prompt = sample.get_field(field_name)
            
            # Extract metadata from sample if available
            metadata = None
            if sample is not None and hasattr(sample, 'metadata'):
                metadata = sample.metadata
            
            batch_item = {"filepath": filepath, "prompt": prompt, "metadata": metadata}
        
        results = self.predict_all([batch_item], samples=[sample] if sample else None)
        return results[0]
    
    def predict_all(self, batch, samples=None):
        """Batch video inference.
        
        Args:
            batch: List of dicts from GetItem with "filepath" and "prompt" keys
            samples: Optional list of FiftyOne samples (for metadata)
        
        Returns:
            List of label dicts (one per video)
        """
        if not batch:
            return []
        
        # Lazy load model
        if self._model is None:
            self._load_model()
        
        results = []
        for i, item in enumerate(batch):
            video_path = item["filepath"]
            prompt_from_batch = item.get("prompt")
            metadata_from_batch = item.get("metadata")
            
            # Check if metadata is required for this operation
            needs_metadata = self.config.operation in [
                "temporal_localization", "comprehensive"
            ]
            
            if needs_metadata and not metadata_from_batch:
                raise ValueError(
                    f"Operation '{self.config.operation}' requires video metadata for timestamp conversion. "
                    f"Please call dataset.compute_metadata() before applying the model."
                )
            
            # Get resolved prompt (before template formatting) for label
            resolved_prompt = prompt_from_batch if prompt_from_batch is not None else self.default_prompt
            if resolved_prompt is None:
                resolved_prompt = "object"  # Fallback for label
            
            # Get full prompt with template for inference
            prompt = self._get_prompt(prompt=prompt_from_batch)
            
            # Run inference
            generated_text, video_metadata = self._run_inference(video_path, prompt)
            logger.debug(f"Generated text: {generated_text[:200]}...")
            
            # Parse output with resolved prompt as label and metadata
            labels = self._parse_output(generated_text, video_metadata, metadata_from_batch, label=resolved_prompt)
            results.append(labels)
        
        return results