from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
import base64
import io
from model.inference import load_model, generate_images
from fastapi import Body

app = FastAPI()

@app.on_event("startup")
def startup_event():
    """
    Load the model and apply LoRA once at server startup.
    Keeps the pipeline warm and avoids re-loading per request.
    """
    load_model()


@app.post("/generate-logo")
async def generate_logo(
    # Request body parameters (JSON). Keep them explicit for clear API schema.
    prompt: str = Body(..., description="Positive prompt describing the desired logo"),
    negative_prompt: str = Body("", description="Optional negative terms to avoid (e.g., 'photo, 3d, text, watermark')"),
    num_images: int = Body(1, description="Number of images to generate for this prompt"),
    height: int = Body(1024, description="Output image height"),
    width: int = Body(1024, description="Output image width")
):
    """
    Generate logo images via the underlying FLUX + LoRA pipeline and return Base64-encoded PNGs.

    Returns:
        JSON:
            {
              "images": ["<BASE64_PNG>", ...]
            }
    Error:
        JSON with HTTP 500 containing {"error": "<message>"} if generation fails.
    """
    try:
        images = generate_images(prompt, negative_prompt, num_images, height, width)
        encoded_images = []

        # Convert PIL images to Base64 PNG strings for transport in JSON
        for img in images:
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_bytes = buffer.getvalue()
            encoded = base64.b64encode(img_bytes).decode("utf-8")
            # If your frontend expects a data URI, uncomment the next line and remove the one above.
            # encoded = "data:image/png;base64," + base64.b64encode(img_bytes).decode("utf-8")
            encoded_images.append(encoded)

        return JSONResponse(content={"images": encoded_images})
    except Exception as e:
        # Keep error surface simple. For production, consider structured logging instead.
        return JSONResponse(status_code=500, content={"error": str(e)})
