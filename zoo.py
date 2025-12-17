"""
FiftyOne integration for Molmo2 video understanding model.

This module provides a batching-enabled implementation of the Molmo2 model
for video understanding tasks in FiftyOne.

Operations:
- pointing: Point to objects in video → frame-level fo.Keypoints
- tracking: Track objects across frames → frame-level fo.Keypoints with fo.Instance
- describe: Text output (description, QA, captioning) → sample-level string

Output Types:
- Frame-level: fo.Keypoints (for pointing and tracking)
- Sample-level: Plain text string (for describe)

Key Design Principles:
- Uses SupportsGetItem + TorchModelMixin for efficient batching
- Operation determines both prompt and parsing behavior
- Timestamps in seconds → FiftyOne 1-based frame numbers
"""

import logging
import re
from typing import Dict, List, Optional, Any, Tuple

import torch

import fiftyone as fo
import fiftyone.core.labels as fol
import fiftyone.core.models as fom
import fiftyone.utils.torch as fout
from fiftyone.core.models import SupportsGetItem, TorchModelMixin
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


# =============================================================================
# Prompt templates
# =============================================================================

OPERATION_PROMPTS = {
    "pointing": "Point to the {target}.",
    "tracking": "Track the {target}.",
    "describe": None,  # User provides prompt directly
}


# =============================================================================
# GetItem class for DataLoader
# =============================================================================

class Molmo2GetItem(GetItem):
    """GetItem transform for Molmo2 video model.
    
    Returns video filepath for processing in predict_all().
    Heavy video processing happens on GPU, not in DataLoader workers.
    """
    
    @property
    def required_keys(self):
        """Fields needed from each sample."""
        return ["filepath"]
    
    def __call__(self, sample_dict):
        """Return video filepath for processing.
        
        Args:
            sample_dict: Dict with "filepath" key
        
        Returns:
            str: Video file path
        """
        return sample_dict["filepath"]


# =============================================================================
# Model Configuration
# =============================================================================

class Molmo2VideoModelConfig(fout.TorchImageModelConfig):
    """Configuration for Molmo2 video model.
    
    Operations:
        - "pointing": Point to objects → fo.Keypoints
        - "tracking": Track objects → fo.Keypoints with fo.Instance
        - "describe": Text output → string field
    
    Key Parameters:
        model_path: HuggingFace model ID (default: "allenai/Molmo2-4B")
        operation: One of "pointing", "tracking", "describe" (default: "pointing")
        target: What to point to/track (set before inference for pointing/tracking)
        prompt: Text prompt (set before inference for describe operation)
        max_new_tokens: Maximum tokens to generate (default: 2048)
    
    Note:
        Parameters can be set after instantiation via model properties.
        Validation happens at inference time, not at instantiation.
    """
    
    def __init__(self, d):
        d["raw_inputs"] = True  # Model handles preprocessing
        super().__init__(d)
        
        # Model path
        self.model_path = self.parse_string(d, "model_path", default="allenai/Molmo2-4B")
        
        # Operation: "pointing", "tracking", or "describe"
        self.operation = self.parse_string(d, "operation", default="pointing")
        
        # Target for pointing/tracking operations (can be set later)
        self.target = self.parse_string(d, "target", default=None)
        
        # Prompt for describe operation (can be set later)
        self.prompt = self.parse_string(d, "prompt", default=None)
        
        # Generation parameters
        self.max_new_tokens = self.parse_number(d, "max_new_tokens", default=2048)


# =============================================================================
# Main Model Class
# =============================================================================

class Molmo2VideoModel(fom.Model, SupportsGetItem, TorchModelMixin):
    """FiftyOne wrapper for Molmo2 video understanding model.
    
    This model processes videos using operation-driven parsing with
    efficient batching via SupportsGetItem + TorchModelMixin.
    
    Operations and Outputs:
        - pointing: fo.Keypoints per frame (points at objects)
        - tracking: fo.Keypoints per frame with fo.Instance linking
        - describe: Plain text string (sample-level)
    
    Batching Architecture:
        - GetItem returns filepath (lightweight)
        - collate_fn returns list of filepaths (no stacking)
        - predict_all processes videos on GPU
    """
    
    def __init__(self, config):
        # Initialize SupportsGetItem (NOT SamplesMixin!)
        SupportsGetItem.__init__(self)
        
        # REQUIRED: Preprocessing flag
        self._preprocess = False
        
        # Store configuration
        self.config = config
        
        # Detect best available device
        self.device = get_device()
        logger.info(f"Using device: {self.device}")
        
        # Lazy-loaded model components
        self._processor = None
        self._model = None
        
        # Fields needed from samples (for SamplesMixin compatibility)
        self._fields = {}
    
    @property
    def needs_fields(self):
        """Dict mapping model-specific keys to sample field names.
        
        This allows the model to access sample-level fields during inference.
        For example, to use a per-sample prompt field:
            model.needs_fields = {"prompt_field": "my_prompts"}
        """
        return self._fields
    
    @needs_fields.setter
    def needs_fields(self, fields):
        """Set the fields this model needs from samples."""
        self._fields = fields
    
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
        valid_ops = ["pointing", "tracking", "describe"]
        if value not in valid_ops:
            raise ValueError(f"operation must be one of {valid_ops}, got '{value}'")
        self.config.operation = value
    
    @property
    def target(self):
        """Target for pointing/tracking operations."""
        return self.config.target
    
    @target.setter
    def target(self, value):
        """Set target for pointing/tracking operations."""
        self.config.target = value
    
    @property
    def prompt(self):
        """Prompt for describe operation."""
        return self.config.prompt
    
    @prompt.setter
    def prompt(self, value):
        """Set prompt for describe operation."""
        self.config.prompt = value
    
    @property
    def max_new_tokens(self):
        """Maximum tokens to generate."""
        return self.config.max_new_tokens
    
    @max_new_tokens.setter
    def max_new_tokens(self, value):
        """Set maximum tokens to generate."""
        self.config.max_new_tokens = value
    
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
        """Factory for GetItem instances."""
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
    # Prompt building
    # =========================================================================
    
    def _get_prompt(self, sample=None) -> str:
        """Get prompt for current operation.
        
        Validates that required parameters are set.
        For describe operation, supports per-sample prompts via needs_fields.
        
        Args:
            sample: Optional FiftyOne sample for per-sample prompt lookup
        
        Raises:
            ValueError: If required parameters are not set for the operation
        """
        if self.config.operation == "describe":
            # Check for per-sample prompt override via needs_fields
            prompt = self.config.prompt
            if sample and self._fields:
                prompt_field = self._fields.get("prompt_field") or next(iter(self._fields.values()), None)
                if prompt_field:
                    field_value = sample.get_field(prompt_field)
                    if field_value:
                        prompt = str(field_value)
            
            if not prompt:
                raise ValueError(
                    "prompt is required for operation='describe'. "
                    "Set it via model.prompt = 'your prompt' or use needs_fields for per-sample prompts"
                )
            return prompt
        else:
            # pointing or tracking
            # Check for per-sample target override via needs_fields
            target = self.config.target
            if sample and self._fields:
                target_field = self._fields.get("target_field") or self._fields.get("target")
                if target_field:
                    field_value = sample.get_field(target_field)
                    if field_value:
                        target = str(field_value)
            
            if not target:
                raise ValueError(
                    f"target is required for operation='{self.config.operation}'. "
                    f"Set it via model.target = 'object to find' or use needs_fields for per-sample targets"
                )
            template = OPERATION_PROMPTS[self.config.operation]
            return template.format(target=target)
    
    def _build_messages(self, video_path: str, sample=None) -> List[Dict]:
        """Build message structure for Molmo2 inference.
        
        Args:
            video_path: Path to video file
            sample: Optional FiftyOne sample for per-sample prompt/target lookup
        """
        prompt = self._get_prompt(sample=sample)
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
    
    def _run_inference(self, video_path: str, sample=None) -> Tuple[str, Dict]:
        """Run model inference on a single video.
        
        Args:
            video_path: Path to video file
            sample: Optional FiftyOne sample for per-sample prompt/target lookup
        
        Returns:
            Tuple of (generated_text, video_metadata)
        """
        messages = self._build_messages(video_path, sample=sample)
        
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
                max_new_tokens=self.config.max_new_tokens
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
    
    def _parse_pointing_output(
        self, 
        text: str, 
        video_metadata: Dict, 
        video_fps: float
    ) -> Dict:
        """Parse <points> output to frame-level fo.Keypoints.
        
        Args:
            text: Model output text
            video_metadata: Dict with "width" and "height"
            video_fps: Video frame rate for timestamp conversion
        
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
                    label=self.config.target or "object",
                    points=[(kp["x"], kp["y"])],
                    index=kp["idx"],
                )
                keypoints_list.append(keypoint)
            
            result[frame_num] = {"points": fol.Keypoints(keypoints=keypoints_list)}
        
        # Add count as sample-level field
        result["count"] = max_idx + 1
        
        return result
    
    def _parse_tracking_output(
        self, 
        text: str, 
        video_metadata: Dict, 
        video_fps: float
    ) -> Dict:
        """Parse <tracks> output to frame-level fo.Keypoints with fo.Instance linking.
        
        Args:
            text: Model output text
            video_metadata: Dict with "width" and "height"
            video_fps: Video frame rate for timestamp conversion
        
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
                label=self.config.target or "tracked",
                points=[(x_norm, y_norm)],
                instance=track_instances[idx],  # Link to track instance
                index=idx,
            )
            
            if frame_num not in frame_labels:
                frame_labels[frame_num] = {"tracks": fol.Keypoints(keypoints=[])}
            
            frame_labels[frame_num]["tracks"].keypoints.append(keypoint)
        
        return frame_labels
    
    def _parse_output(self, text: str, video_metadata: Dict, sample) -> Dict:
        """Parse model output based on operation type.
        
        Args:
            text: Model output text
            video_metadata: Dict with "width" and "height"
            sample: FiftyOne sample for metadata
        
        Returns:
            Dict with sample-level (string keys) and frame-level (int keys) labels
        """
        # Get video FPS from sample metadata
        video_fps = 30.0  # Default
        if hasattr(sample, 'metadata') and sample.metadata:
            if hasattr(sample.metadata, 'frame_rate') and sample.metadata.frame_rate:
                video_fps = sample.metadata.frame_rate
        
        if self.config.operation == "describe":
            # Plain text output
            return {"description": text}
        
        elif self.config.operation == "pointing":
            # Check if output has grounded coordinates
            if has_grounded_output(text):
                return self._parse_pointing_output(text, video_metadata, video_fps)
            else:
                # Model returned text instead of coordinates
                return {"description": text}
        
        elif self.config.operation == "tracking":
            if has_grounded_output(text):
                return self._parse_tracking_output(text, video_metadata, video_fps)
            else:
                return {"description": text}
        
        return {"description": text}
    
    # =========================================================================
    # Main inference methods
    # =========================================================================
    
    def predict(self, arg, sample=None):
        """Single video inference.
        
        Args:
            arg: Video filepath (from GetItem)
            sample: FiftyOne sample (for metadata)
        
        Returns:
            Dict with labels
        """
        results = self.predict_all([arg], samples=[sample] if sample else None)
        return results[0]
    
    def predict_all(self, video_paths, samples=None):
        """Batch video inference.
        
        Args:
            video_paths: List of video filepaths from GetItem
            samples: Optional list of FiftyOne samples (for metadata and per-sample fields)
        
        Returns:
            List of label dicts (one per video)
        """
        # Lazy load model
        if self._model is None:
            self._load_model()
        
        results = []
        for i, video_path in enumerate(video_paths):
            sample = samples[i] if samples else None
            
            try:
                # Run inference (pass sample for per-sample prompt/target lookup)
                generated_text, video_metadata = self._run_inference(video_path, sample=sample)
                logger.debug(f"Generated text: {generated_text[:200]}...")
                
                # Parse output
                labels = self._parse_output(generated_text, video_metadata, sample)
                results.append(labels)
                
            except Exception as e:
                logger.error(f"Error processing {video_path}: {e}")
                results.append({"error": str(e)})
        
        return results

