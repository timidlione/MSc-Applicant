import os
import torch
from diffusers import FluxPipeline
from safetensors.torch import load_file

pipe = None  # Global pipeline object (initialized once on app startup)

def find_flux_model_attr(pipeline):
    """
    Find the internal FluxTransformer2DModel inside the FLUX pipeline.

    Returns:
        (attr_name, module): attribute name on the pipeline and the module itself.

    Raises:
        AttributeError: if the transformer module cannot be found (version/key drift).
    """
    for name, module in vars(pipeline).items():
        if module.__class__.__name__ == "FluxTransformer2DModel":
            return name, module
    raise AttributeError("FluxTransformer2DModel component not found in the pipeline.")

def apply_lora(module, lora_path):
    """
    Apply LoRA weights (safetensors) to the given module.

    Args:
        module: target module to receive LoRA weights.
        lora_path (str): path to .safetensors file.

    Notes:
        - Uses strict=False to allow partial key matches across versions.
        - Keep lora_path outside of version control if it’s large or private.
    """
    print(f"Applying LoRA: {lora_path}")
    state_dict = load_file(lora_path)
    module.load_state_dict(state_dict, strict=False)
    print("LoRA applied.")

def load_model():
    """
    Load base FLUX.1-dev pipeline, move it to device, then apply LoRA.

    Device/Dtype:
        - Uses CUDA and bfloat16 by default (good for A100/H100).
        - If dtype or device issues occur, consider switching to float16 or CPU/MPS.
    """
    global pipe
    print("Loading base model...")
    base_model_id = "black-forest-labs/FLUX.1-dev"
    lora_path = "./downloaded_lora/pytorch_lora_weights.safetensors"  # expected local LoRA path

    pipe = FluxPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16
    ).to("cuda")

    # Find the internal transformer and apply LoRA onto it
    attr_name, flux_model = find_flux_model_attr(pipe)
    apply_lora(flux_model, lora_path)
    setattr(pipe, attr_name, flux_model)

    print("Model and LoRA loaded.")

def generate_images(prompt: str, negative_prompt: str = "", num_images: int = 1, height: int = Body(1024), width: int = Body(1024)):
    """
    Generate images with the loaded FLUX pipeline + LoRA.

    Args:
        prompt (str): positive text prompt describing the desired logo.
        negative_prompt (str): optional negative terms to avoid (e.g., 'photo, 3d, watermark').
        num_images (int): number of images to generate per request.
        height (int): output image height.
        width (int): output image width.

    Returns:
        List[PIL.Image.Image]: generated images.

    Important:
        - This function references FastAPI's Body(...) in defaults; if used outside a FastAPI router,
          ensure Body is imported from fastapi or replace with plain integers.
        - Typical quality knobs: num_inference_steps (quality vs. speed), guidance_scale (prompt adherence).
    """
    global pipe
    results = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=100,  # ↑ for quality, ↓ for speed
        guidance_scale=10,        # higher → more prompt adherence, risk of over-constrain
        height=height,
        width=width,
        num_images_per_prompt=num_images
    )
    return results.images
