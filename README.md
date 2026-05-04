# 🌊 UnderWater-Sim: Underwater Image Simulation & Restoration

> A web-based application implementing the UnderWater-Sim algorithm for underwater image simulation and physics-based restoration.

---

## 📌 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Quick Start](#quick-start)
- [Manual Setup (No Docker)](#manual-setup-no-docker)
- [API Reference](#api-reference)
- [Pipeline Details](#pipeline-details)
- [Project Structure](#project-structure)
- [Research Notes](#research-notes)

---

## Overview

This project implements two complementary pipelines:

| Mode | Input | Output |
|------|-------|--------|
| **Simulate** | Clean RGB image | Degraded underwater image |
| **Restore** | Degraded underwater image | Recovered scene radiance |

Both pipelines are grounded in the **underwater image formation model**:

```
I(x) = J(x) · t(x) + B(x) · (1 - t(x))
```

Where:
- `I(x)` — captured underwater image
- `J(x)` — true scene radiance (desired output)
- `t(x)` — transmission map
- `B(x)` — atmospheric light / backscatter

---

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Browser Client                     │
│   ┌──────────┐  ┌──────────┐  ┌────────────────┐   │
│   │  Image   │  │  Video   │  │    Webcam      │   │
│   │  Upload  │  │  Upload  │  │  Live Preview  │   │
│   └────┬─────┘  └────┬─────┘  └───────┬────────┘   │
│        └─────────────┴────────────────┘             │
│                       │                             │
│            HTTP/multipart POST                      │
└───────────────────────┼─────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────┐
│              FastAPI Backend (port 8000)             │
│                                                      │
│   ┌─────────────────────┐  ┌──────────────────────┐ │
│   │   Simulate Pipeline │  │  Restore Pipeline    │ │
│   │                     │  │                      │ │
│   │  color_shift        │  │  dark_channel        │ │
│   │  haze               │  │  transmission        │ │
│   │  depth_blur         │  │  recover_scene       │ │
│   │  light_rays         │  │  CLAHE post-proc     │ │
│   │  particles          │  │                      │ │
│   │  wave_distortion    │  └──────────────────────┘ │
│   │  vignette           │                           │
│   └─────────────────────┘                           │
└─────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites
- [Docker](https://www.docker.com/) and Docker Compose installed

### Run with Docker

```bash
# Clone / navigate to the project
cd UnderWater-Sim-app

# Build and start both services
docker-compose up --build

# App will be available at:
# Frontend → http://localhost:3000
# Backend API → http://localhost:8000
# API Docs → http://localhost:8000/docs
```

To stop:
```bash
docker-compose down
```

---

## Manual Setup (No Docker)

### Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

The frontend is a single HTML file — no build step needed.

```bash
# Option 1: Open directly in browser
open frontend/index.html

# Option 2: Serve with Python
cd frontend
python -m http.server 3000
# Then visit http://localhost:3000
```

> ⚠️ If opening `index.html` directly (file://), webcam and fetch requests may be blocked by CORS. Use Python's HTTP server instead.

---

## API Reference

Full interactive docs available at `http://localhost:8000/docs` (Swagger UI).

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Server health check |
| `POST` | `/simulate/image` | Simulate underwater degradation on an image |
| `POST` | `/restore/image` | Restore an underwater image |
| `POST` | `/simulate/video` | Process a full video (simulate) |
| `POST` | `/restore/video` | Process a full video (restore) |
| `POST` | `/webcam/frame` | Process a single webcam frame |

### Example: Simulate an Image

```bash
curl -X POST http://localhost:8000/simulate/image \
  -F "file=@my_photo.jpg" \
  -F 'params={"haze_strength": 0.2, "ray_intensity": 0.05}'
```

Response:
```json
{
  "original": "<base64 jpeg>",
  "result":   "<base64 jpeg>",
  "mode":     "simulate"
}
```

### Simulate Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `haze_strength` | `0.15` | 0–0.5 | Backscatter / haze intensity |
| `blur_strength` | `0.50` | 0–1.0 | Depth-based blur at bottom |
| `ray_intensity` | `0.03` | 0–0.15 | Light caustic ray brightness |
| `wave_amplitude`| `6.0` | 0–20 | Surface refraction amplitude |
| `particle_density` | `1.0` | 0–3.0 | Suspended particle count |

### Restore Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `omega` | `0.95` | 0.5–1.0 | Haze removal aggressiveness |
| `t0` | `0.25` | 0.1–0.5 | Minimum transmission clamp |

---

## Pipeline Details

### Simulation Pipeline

```
Input Image
    │
    ▼
color_shift()       ← Red×0.75, Green×0.9, Blue×1.05
    │
    ▼
apply_haze()        ← Gradient backscatter (top→bottom)
    │
    ▼
depth_blur()        ← Progressive Gaussian blur with depth
    │
    ▼
apply_light_rays()  ← Caustic ray simulation (seeded for video)
    │
    ▼
add_particles()     ← Random suspended particulate (seeded)
    │
    ▼
apply_wave_distortion() ← Sinusoidal remap (surface refraction)
    │
    ▼
apply_vignette()    ← Gaussian lens falloff
    │
    ▼
Output (Simulated Underwater Image)
```

### Restoration Pipeline

```
Underwater Image
    │
    ▼
dark_channel()              ← Dark Channel Prior (15×15 patch)
    │
    ▼
estimate_atmospheric_light() ← Top 0.1% bright pixels in dark channel
    │
    ▼
estimate_transmission()      ← t(x) = 1 - ω × dark_channel(I/A)
    │
    ▼
recover_scene()              ← J(x) = (I(x) - A) / max(t(x), t0) + A
    │
    ▼
CLAHE post-processing        ← Contrast-limited adaptive histogram eq.
    │
    ▼
Output (Restored Image)
```

---

## Project Structure

```
UnderWater-Sim-app/
│
├── docker-compose.yml
│
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py               ← FastAPI server & all endpoints
│   ├── pipeline.py           ← simulate_pipeline / restore_pipeline
│   │
│   ├── underwater/           ← Physics model modules
│   │   ├── __init__.py
│   │   ├── dark_channel.py
│   │   ├── transmission.py
│   │   └── recover_scene.py
│   │
│   ├── filters/              ← Visual effect filters
│   │   ├── __init__.py
│   │   ├── color_shift.py
│   │   ├── haze.py
│   │   ├── depth_blur.py
│   │   ├── light_rays.py
│   │   ├── particles.py
│   │   ├── waves.py
│   │   └── vignette.py
│   │
│   └── utils/
│       ├── __init__.py
│       └── io_utils.py
│
└── frontend/
    ├── Dockerfile
    └── index.html            ← Complete single-file React-free UI
```

---

## Research Notes

### Bugs Fixed from Original Codebase

| File | Bug | Fix Applied |
|------|-----|-------------|
| `particles.py` | uint8 overflow — adding to bright pixels wrapped to dark values | Cast to int32 before addition, clip then cast back |
| `haze.py` | Gradient direction reversed — was dimming top instead of bottom | Flipped `np.linspace` direction + added backscatter tint |
| `depth_blur.py` | In-place row mutation — later rows blended against partially modified image | Work on a copy before writing back |
| `transmission.py` | `from underwater.dark_channel import ...` breaks inside package | Changed to relative import: `from .dark_channel import ...` |
| `recover_scene.py` | No uint8 normalization guard — scene recovery math on raw uint8 | Explicit float32 cast before computation |
| `light_rays.py` | No random seed — causes flickering on every video frame | Added optional `seed` parameter |
| All modules | Missing `__init__.py` — package imports all fail | Added `__init__.py` to all packages |

### Key References

- He, K., Sun, J., & Tang, X. (2011). *Single image haze removal using dark channel prior.* IEEE TPAMI.
- Akkaynak, D., & Treibitz, T. (2019). *UnderWater-Sim: A method for removing water from underwater images.* CVPR.

---

## License

MIT License. See `LICENSE` for details.
