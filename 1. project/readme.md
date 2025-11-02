# Generative AI Web Service for Logo Creation & Branding (FLUX + LoRA)

FastAPI-based web API that generates **logo images from text prompts** using **FLUX.1-dev + LoRA**.
Send a prompt to `/generate-logo` and receive **Base64-encoded PNG** images.
 
> **Scope**
>
> * This repo is **inference-only** (no training code or datasets).
> * LoRA weights can be loaded from a **local file** or **Hugging Face**.
> * Built and tested on **RunPod GPU instances (NVIDIA A100/H100, CUDA 12.1)** with **VS Code Remote-SSH**.

---

## 1) Project Structure

```
project/
├── app.py                        # FastAPI server (startup + /generate-logo)
├── model/
│   ├── __init__.py
│   └── inference.py              # FLUX pipeline load + LoRA apply + inference
├── downloaded_lora/
│   └── pytorch_lora_weights.safetensors
├── outputs/                      # optional: save generated images (not required)
├── assets/
│   └── samples/
│      ├── logo1.png             # sample result (add later)
│      └── logo2.png             # sample result (add later)
└── README.md
```

> Current code expects a **local LoRA file** at `downloaded_lora/pytorch_lora_weights.safetensors`.

---

## 2) Environment & Requirements

* **Platform:** RunPod (Linux) on **NVIDIA A100/H100**
* **CUDA / PyTorch:** CUDA 12.1 wheels
* **Editor:** **VS Code Remote-SSH**
* **Core libs:** FastAPI, Uvicorn, Diffusers, Transformers, Accelerate, PyTorch, Safetensors

### Pinned versions used in this project

* PyTorch **2.5.1+cu121** / torchvision **0.20.1+cu121** / torchaudio **2.5.1**
* diffusers **0.35.1**
* transformers **4.45.2**
* accelerate **1.1.1**
* safetensors **0.4.5**
* huggingface_hub (latest) — only needed if you download weights from HF at runtime
* sentencepiece (needed for some tokenizers; safe to include)
* Pillow, numpy

### Install (exact commands)

```bash
# 0) API server
pip install "uvicorn[standard]" fastapi python-multipart

# 1) PyTorch CUDA 12.1 wheels (A100/H100)
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121

# 2) Diffusers stack
pip install diffusers==0.35.1 transformers==4.45.2 accelerate==1.1.1 safetensors==0.4.5

# 3) Utilities
pip install pillow "numpy>=1.24,<3" sentencepiece

# 4) (Optional) Hugging Face Hub for remote LoRA
pip install huggingface_hub
# If you need private access:
# huggingface-cli login    # <<< Do NOT commit tokens; use environment variables or CI secrets.
```

---

## 3) How to Run (RunPod/CUDA + VS Code)

1. Place LoRA at:

   ```
   downloaded_lora/pytorch_lora_weights.safetensors
   ```

2. Launch API:

   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

> First run downloads the base model; loading may take time.

---

## 4) API

### `POST /generate-logo`

**Request (JSON)**

```json
{
  "prompt": "minimal cute fox mascot logo, flat vector, centered, white background",
  "negative_prompt": "photo, 3d, text, watermark, low quality",
  "num_images": 1,
  "height": 1024,
  "width": 1024
}
```

**Response (JSON)**

```json
{
  "images": ["<BASE64_PNG>", "<BASE64_PNG>"]
}
```

> **Note:** The API returns **raw Base64 strings** (no `data:image/png;base64,` prefix).
> If you need a data URI for `<img src>`, prepend that prefix on the client side.

**cURL**

```bash
curl -X POST http://localhost:8000/generate-logo \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "luxury monogram logo, gold foil, black background, geometric, centered",
    "negative_prompt": "photo, 3d, noisy, watermark, low quality",
    "num_images": 1, "height": 1024, "width": 1024
  }' | jq '.images[0]' -r | sed 's/"//g' | base64 --decode > out.png
```

---

## 5) How It Works (Code Overview)

* `app.py`

  * Startup loads FLUX + applies LoRA once.
  * `/generate-logo` → `generate_images(...)` → PIL → Base64 PNG → JSON.

* `model/inference.py`

  * Loads **FLUX.1-dev** → moves to **CUDA** → finds `FluxTransformer2DModel` → loads LoRA `state_dict` (`strict=False`).
  * Default params: `num_inference_steps=100`, `guidance_scale=10`, size from request, `num_images_per_prompt`.

> You can switch to `pipe.load_lora_weights(...)` depending on Diffusers API version.

---

## 6) LoRA Weights

All LoRA weights are hosted on Hugging Face under our organization:

👉 [https://huggingface.co/logologolab](https://huggingface.co/logologolab)

> This repo does **not** store `.safetensors` files.
> Download the desired LoRA from the link above and either:
>
> * **Use at runtime** with `huggingface_hub.hf_hub_download`, or
> * **Pre-download** and point `inference.py` to the local file path.

---

## 7) Prompting Quick Guide

* **Style tokens** (must include user-chosen style):
  `simple`, `minimal`, `retro`, `vintage`, `cute`, `playful`, `luxury`, `tattoo`, `futuristic`, `cartoon`, `watercolor`

* **Logo types**:

  * Image + Text → `icon with text`
  * Text-only → `text-only design`
  * Image-only → `icon only`

* **Defaults**: Steps **100**, Guidance **10**, Size **1024**, Background `white background`,
  Negative: `photo, 3d, text, watermark, low quality, noisy`

* **Style snippets**

  * minimal: `white background, simple black icon, clean lines, minimal logo, modern sans-serif font`
  * vintage: `retro vintage logo, distressed texture, old-school serif font, classic badge design`
  * cute/playful: `cute playful logo, colorful cartoon mascot, rounded sans-serif font, cheerful design`
  * luxury: `luxury_premium_logo_lora`
  * tattoo: `old-school tattoo style, bold black outlines, traditional Americana motifs, intricate linework, vintage tattoo aesthetic`
  * futuristic: `futuristic logo, sleek metallic surfaces, neon glow accents, holographic effects, modern techno font, digital circuit-inspired design`
  * cartoon: `a cartoon-style logo of a cute animal, vector, colorful, minimal design`
  * watercolor: `watercolor, hand-drawn, soft tones, pastel colors, textured brush strokes, natural flow, light ink wash, artistic feel`

> Keep the final prompt under **77 tokens** when possible.

---

## 8) Sample Results

Below are two sample outputs generated by this API + LoRA (examples; not affiliated with any real brand):

<p align="center">
  <img src="assets/samples/logo1.png" alt="Sample Logo 1" width="360" />
  &nbsp;&nbsp;&nbsp;
  <img src="assets/samples/logo2.png" alt="Sample Logo 2" width="360" />
</p>
<p align="center">
  <sub><strong>Logo 1</strong>: futuristic style sample &nbsp;|&nbsp; <strong>Logo 2</strong>: tattoo style sample</sub>
</p>

---

## 9) Notes, Security & Troubleshooting

* **Security:** never commit secrets (e.g., `HUGGINGFACE_HUB_TOKEN`). Use env vars or CI secrets.
* **CUDA OOM:** reduce size/steps/`num_images`.
* **Photographic look:** strengthen vector terms (`flat vector`, `line-art`) + negatives (`photo, 3d`).
* **Dtype issues:** swap `bfloat16` ↔ `float16` depending on your CUDA build.

---

좋아, **Code license** 문구만 확실히 정리해서 교체본 드릴게. 아래 블록을 README의 **## 10) License & Credits** 섹션에 그대로 붙여 넣으면 돼.

---

## 10) License & Credits

* **Base model**: `black-forest-labs/FLUX.1-dev` (respect its license/usage terms).
* **LoRA**: trained by our team (open-source distribution allowed; dataset not redistributed).
* **Code license**: **MIT License** — add a `LICENSE` file and include the SPDX header in source files.

**LoRA Weights:** hosted at [https://huggingface.co/logologolab](https://huggingface.co/logologolab) (see each model card for license/usage terms).
**Trademark/Copyright:** Do not generate deceptive or infringing marks. Use responsibly.

**LICENSE (MIT) template**

```
MIT License

Copyright (c) 2025 Sangmin Woo and contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
