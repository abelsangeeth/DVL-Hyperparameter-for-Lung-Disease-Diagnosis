"""FastAPI service around an exported bundle.

    dvlhg serve --bundle runs/export --port 8000

Serves the single-page frontend at / and the JSON API under /api. The model is
loaded once on first use, not at import, so the process starts instantly and a
missing bundle produces a clear 503 rather than a crash at boot.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..constants import DISCLAIMER, LABELS
from ..utils import get_logger

LOG = get_logger()

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

_predictor = None
_load_error: Optional[str] = None


def frontend_dir() -> Optional[Path]:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "frontend"
        if (candidate / "index.html").exists():
            return candidate
    return None


def get_predictor():
    """Load the bundle on first request and cache it."""
    global _predictor, _load_error
    if _predictor is not None:
        return _predictor
    if _load_error is not None:
        raise HTTPException(status_code=503, detail=_load_error)
    from .inference import Predictor

    bundle = os.environ.get("DVLHG_BUNDLE", "")
    if not bundle:
        _load_error = "DVLHG_BUNDLE is not set — start the server with `dvlhg serve --bundle <dir>`"
        raise HTTPException(status_code=503, detail=_load_error)
    try:
        _predictor = Predictor(
            bundle,
            device=os.environ.get("DVLHG_DEVICE", "auto"),
            neighbours=int(os.environ.get("DVLHG_NEIGHBOURS", "6")),
        )
    except Exception as exc:  # noqa: BLE001
        _load_error = f"could not load the bundle at {bundle}: {exc}"
        LOG.error(_load_error)
        raise HTTPException(status_code=503, detail=_load_error) from exc
    return _predictor


async def _read_image(upload: UploadFile) -> bytes:
    """Validate and read an uploaded film. Shared by /predict and /refine."""
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix and suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported image type '{suffix}'; expected one of {sorted(ALLOWED_SUFFIXES)}",
        )
    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413, detail=f"image larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB"
        )
    return data


def create_app(allow_origins=("*",)) -> FastAPI:
    app = FastAPI(
        title="DVL-HGN — chest X-ray finding support",
        description=DISCLAIMER,
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(allow_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "bundle": os.environ.get("DVLHG_BUNDLE", ""),
            "model_loaded": _predictor is not None,
            "labels": LABELS,
            "disclaimer": DISCLAIMER,
        }

    @app.get("/api/model-card")
    def model_card():
        return get_predictor().model_card()

    @app.post("/api/predict")
    async def predict(
        image: UploadFile = File(...),
        report: str = Form(""),
        explain: bool = Form(True),
        neighbours: int = Form(6),
    ):
        data = await _read_image(image)
        predictor = get_predictor()
        try:
            result = predictor.predict(
                data, report=report or "", explain=bool(explain), neighbours=int(neighbours)
            )
        except Exception as exc:  # noqa: BLE001
            LOG.exception("prediction failed")
            raise HTTPException(status_code=400, detail=f"could not process this image: {exc}") from exc
        return JSONResponse(result)

    @app.get("/api/capabilities")
    def capabilities():
        """What the UI may offer. Cheap enough to call on page load."""
        predictor = get_predictor()
        return {
            "labels": LABELS,
            "diffusion": predictor.diffusion_available(),
            "diffusion_info": predictor.diffusion_info(),
            "explain": True,
            "vlm_panel": True,
            "serving_variant": predictor.meta.get("serving_variant"),
        }

    @app.post("/api/generate")
    def generate(
        atelectasis: bool = Form(False),
        cardiomegaly: bool = Form(False),
        edema: bool = Form(False),
        pleural_effusion: bool = Form(False),
        guidance_scale: float = Form(2.0),
        steps: int = Form(50),
        seed: int = Form(-1),
    ):
        """Sample a synthetic film from the diffusion model."""
        predictor = get_predictor()
        labels = [float(atelectasis), float(cardiomegaly), float(edema), float(pleural_effusion)]
        try:
            return JSONResponse(
                predictor.generate(
                    labels,
                    guidance_scale=max(1.0, min(float(guidance_scale), 8.0)),
                    steps=max(5, min(int(steps), 250)),
                    seed=None if int(seed) < 0 else int(seed),
                )
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            LOG.exception("generation failed")
            raise HTTPException(status_code=500, detail=f"generation failed: {exc}") from exc

    @app.post("/api/refine")
    async def refine(
        image: UploadFile = File(...),
        strength: float = Form(0.25),
        steps: int = Form(20),
    ):
        """SDEdit-refine an uploaded film: noise partway, denoise back."""
        data = await _read_image(image)
        predictor = get_predictor()
        try:
            return JSONResponse(
                predictor.refine(
                    data,
                    strength=max(0.02, min(float(strength), 0.95)),
                    steps=max(5, min(int(steps), 100)),
                )
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            LOG.exception("refinement failed")
            raise HTTPException(status_code=500, detail=f"refinement failed: {exc}") from exc

    static_dir = frontend_dir()
    if static_dir is not None:
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @app.get("/")
        def index():
            return FileResponse(str(static_dir / "index.html"))
    else:
        @app.get("/")
        def index_missing():
            return {"message": "frontend/index.html not found; the API is still available under /api"}

    return app


app = create_app()


def run(bundle: str, host: str = "127.0.0.1", port: int = 8000, device: str = "auto", neighbours: int = 6):
    import uvicorn

    os.environ["DVLHG_BUNDLE"] = str(Path(bundle).resolve())
    os.environ["DVLHG_DEVICE"] = device
    os.environ["DVLHG_NEIGHBOURS"] = str(neighbours)
    LOG.info("serving bundle %s on http://%s:%d", bundle, host, port)
    uvicorn.run(create_app(), host=host, port=port, log_level="info")
