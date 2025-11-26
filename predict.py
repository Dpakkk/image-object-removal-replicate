"""
Replicate predictor for object removal using LaMa inpainting
"""

import io
import base64
import numpy as np
from PIL import Image
from typing import Optional
import torch
from cog import BasePredictor, Input, Path
import tempfile

# LaMa Cleaner imports
from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config, HDStrategy, LDMSampler


class Predictor(BasePredictor):
    def setup(self):
        """Load the LaMa model into memory"""
        print("Loading LaMa model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # Create model manager with LaMa model
        self.model_manager = ModelManager(
            name="lama",
            device=device,
        )

        print(f"✓ LaMa model loaded on {device}")

    def predict(
        self,
        image: Path = Input(description="Input image to remove objects from"),
        mask: Path = Input(description="Binary mask indicating areas to remove (white = remove, black = keep)"),
        hd_strategy_resize_limit: int = Input(
            description="Resize limit for HD strategy. Larger images will be resized to this limit for faster processing.",
            default=800,
            ge=512,
            le=2048
        ),
    ) -> Path:
        """
        Remove objects from an image using LaMa inpainting.

        Args:
            image: Input image file
            mask: Binary mask image (white areas will be removed)
            hd_strategy_resize_limit: Maximum dimension for processing (affects speed/quality tradeoff)

        Returns:
            Path to the inpainted image
        """

        # Load and convert image to RGB numpy array
        input_image = Image.open(image)
        if input_image.mode != "RGB":
            input_image = input_image.convert("RGB")
        image_array = np.array(input_image)

        # Load and convert mask to grayscale
        mask_image = Image.open(mask)
        if mask_image.mode != "L":
            mask_image = mask_image.convert("L")
        mask_array = np.array(mask_image)

        print(f"Processing image: {image_array.shape}, mask: {mask_array.shape}")

        # Ensure mask is binary (0 or 255)
        mask_binary = (mask_array > 127).astype(np.uint8) * 255

        # Create inpainting config
        config = Config(
            ldm_steps=20,
            ldm_sampler=LDMSampler.plms,
            hd_strategy=HDStrategy.RESIZE,
            hd_strategy_crop_margin=32,
            hd_strategy_crop_trigger_size=hd_strategy_resize_limit,
            hd_strategy_resize_limit=hd_strategy_resize_limit,
        )

        # Run inpainting
        result = self.model_manager(
            image=image_array,
            mask=mask_binary,
            config=config
        )

        # Convert result to uint8 if needed
        if result.dtype != np.uint8:
            if result.max() <= 1.0:
                result = (result * 255).astype(np.uint8)
            else:
                result = result.astype(np.uint8)

        # LaMa returns BGR, convert to RGB for PIL
        rgb_result = result[:, :, ::-1]

        # Convert to PIL Image
        output_image = Image.fromarray(rgb_result.astype(np.uint8))

        # Save to temporary file
        output_path = Path(tempfile.mktemp(suffix=".png"))
        output_image.save(output_path, format="PNG", quality=95)

        print("✓ Inpainting completed")

        return output_path
