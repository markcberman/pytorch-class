# pytorch-class

## Model Setup (Important)

This project uses multiple machine learning models that are **not stored in the Git repository** to keep the repo lightweight.

---

## Stable Diffusion Model (Hugging Face - Mirror)

This project uses the mirrored Stable Diffusion model:

Manojb/stable-diffusion-2-base

👉 https://huggingface.co/Manojb/stable-diffusion-2-base

✅ This is a **public mirror**, so:
- No authentication required
- No Hugging Face login needed
- No license acceptance required

---

## Device Compatibility

This project automatically configures the model based on your hardware:

- **CUDA (GPU)** → uses `float16` and `variant="fp16"` (fastest)
- **CPU** → uses `float32`
- **MPS (Apple Silicon)** → uses `float32`

No manual changes are required.

---

## Automatic Model Download

The Stable Diffusion model is automatically:

- downloaded if missing
- cached in `./models`
- reused on subsequent runs

No manual setup is required.

---

## Fruits Classification Model

The file `fruits_quality_model.pth` is also automatically downloaded if missing.

- Stored in `./models`
- Downloaded from GitHub Releases
- Loaded transparently by the code

---

## Notes

- Models are excluded from Git using `.gitignore`
- This keeps the repository:
  - lightweight
  - fast to clone
  - reproducible

Example `.gitignore` entries:

models/
*.pth

---

## Optional Alternative Model

If you prefer, you can switch to a fully public model:

runwayml/stable-diffusion-v1-5

👉 https://huggingface.co/runwayml/stable-diffusion-v1-5
