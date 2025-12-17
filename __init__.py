"""
Molmo2 FiftyOne Model Zoo Integration.

This module provides the Molmo2 video understanding model as a FiftyOne
remote zoo model with efficient batching support.

Operations:
    - pointing: Point to objects in video → frame-level fo.Keypoints
    - tracking: Track objects across frames → frame-level fo.Keypoints with fo.Instance
    - describe: Text output (description, QA, captioning) → sample-level string

Example:
    import fiftyone as fo
    import fiftyone.zoo as foz
    
    # Load model for pointing
    model = foz.load_zoo_model(
        "molmo2",
        operation="pointing",
        target="penguins"
    )
    
    # Apply to video dataset
    dataset.apply_model(model, label_field="molmo2_points")
"""

import logging
import os

from huggingface_hub import snapshot_download
from fiftyone.operators import types

from .zoo import Molmo2VideoModel, Molmo2VideoModelConfig

logger = logging.getLogger(__name__)


def download_model(model_name, model_path):
    """Downloads the Molmo2 model from HuggingFace.

    Args:
        model_name: the name of the model to download, as declared by the
            ``base_name`` and optional ``version`` fields of the manifest
        model_path: the absolute filename or directory to which to download the
            model, as declared by the ``base_filename`` field of the manifest
    """
    snapshot_download(repo_id=model_name, local_dir=model_path)


def load_model(model_name=None, model_path=None, **kwargs):
    """Load a Molmo2 video model for use with FiftyOne.
    
    Args:
        model_name: Model name (unused, for compatibility)
        model_path: HuggingFace model ID or path to model files
            Default: "allenai/Molmo2-4B"
        **kwargs: Additional config parameters:
            - operation: "pointing", "tracking", or "describe" (default: "pointing")
            - target: What to point to/track (required for pointing/tracking)
            - prompt: Text prompt (required for describe operation)
            - max_new_tokens: Max tokens to generate (default: 2048)
        
    Returns:
        Molmo2VideoModel: Initialized model ready for inference
    
    Examples:
        # Pointing to objects
        model = load_model(
            model_path="allenai/Molmo2-4B",
            operation="pointing",
            target="penguins"
        )
        
        # Tracking objects
        model = load_model(
            operation="tracking",
            target="the red car"
        )
        
        # Description/QA/Captioning
        model = load_model(
            operation="describe",
            prompt="Describe this video in detail."
        )
        
        # Apply to dataset
        dataset.apply_model(model, label_field="predictions")
    """
    if model_path is None:
        model_path = "allenai/Molmo2-4B"
    
    config_dict = {"model_path": model_path}
    config_dict.update(kwargs)
    
    config = Molmo2VideoModelConfig(config_dict)
    return Molmo2VideoModel(config)


def resolve_input(model_name, ctx):
    """Defines properties to collect the model's custom parameters.

    Args:
        model_name: the name of the model
        ctx: an ExecutionContext

    Returns:
        a fiftyone.operators.types.Property
    """
    inputs = types.Object()
    
    # Operation selection
    inputs.enum(
        "operation",
        values=["pointing", "tracking", "describe"],
        default="pointing",
        label="Operation",
        description="Type of video analysis: pointing (locate objects), tracking (track across frames), describe (text output)",
    )
    
    # Target for pointing/tracking
    inputs.str(
        "target",
        default=None,
        required=False,
        label="Target",
        description="What to point to or track (required for pointing/tracking operations). Examples: 'penguins', 'the red car', 'all people'",
    )
    
    # Prompt for describe operation
    inputs.str(
        "prompt",
        default=None,
        required=False,
        label="Prompt",
        description="Text prompt for describe operation. Examples: 'Describe this video.', 'What color is the car?'",
    )
    
    # Generation parameters
    inputs.int(
        "max_new_tokens",
        default=2048,
        label="Max New Tokens",
        description="Maximum tokens to generate in response",
    )
    
    return types.Property(inputs)
