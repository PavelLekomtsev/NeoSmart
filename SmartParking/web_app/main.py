"""
Smart Parking Web Application
FastAPI server - receives frames from UE5 via HTTP.
Supports multiple cameras.
"""

import asyncio
import base64
import json
import logging
import os
import pickle
import cv2
import numpy as np
import time
from pathlib import Path
from typing import Optional

from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from neosmart.config import get_settings
from neosmart.logging_setup import configure_logging, print_banner, print_section

from detector import ParkingDetector, create_placeholder_image
from barrier_controller import BarrierController
from barrier_db import BarrierDatabase

_settings = get_settings()
configure_logging(_settings)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Pre-load the detector and barrier controllers at server startup."""
    print_section("Detector")
    get_detector()
    print_section("Barrier System")
    init_barrier_controllers()
    print_section("Ready")
    logger.info("Server ready — dashboard at http://localhost:8000")
    yield


app = FastAPI(
    title="Smart Parking Monitor",
    description="Real-time parking monitoring with AI detection",
    version="2.0.0",
    lifespan=lifespan,
)

BASE_DIR = Path(__file__).parent


class ImmutableStaticFiles(StaticFiles):
    """StaticFiles that tells the browser to cache forever.

    Every reference to a static asset in the HTML uses a `?v=NN` query string
    (e.g. `app.js?v=11`), so when we update an asset we also bump its version
    and the browser fetches the new URL. Marking responses as `immutable` lets
    the browser skip conditional revalidation entirely — no more 304 round-trips
    cluttering the server log on every page load."""

    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return resp


app.mount("/static", ImmutableStaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

FRAMES_DIR = _settings.paths.resolve(_settings.paths.frames_dir)

CAMERA_IDS = _settings.camera_ids

FRAME_PATHS = {cam_id: _settings.frame_path(cam_id) for cam_id in CAMERA_IDS}

BARRIER_CAMERA_IDS = _settings.barrier_camera_ids

# Barrier camera roles
ENTRY_PLATE_CAMERA = _settings.barrier.entry_plate_camera      # Plate recognition for entry barrier
ENTRY_SAFETY_CAMERA = _settings.barrier.entry_safety_camera    # Safety zone monitoring for entry barrier


class FrameStorage:
    def __init__(self):
        self.current_frame: Optional[np.ndarray] = None
        self.last_update: float = 0
        self.frame_count: int = 0

frame_storages = {cam_id: FrameStorage() for cam_id in CAMERA_IDS}

# One shared detector (single YOLO model, used for all cameras)
detector: Optional[ParkingDetector] = None


def get_detector() -> ParkingDetector:
    """Get or create the shared detector instance."""
    global detector
    if detector is None:
        detector = ParkingDetector()
    return detector


# --- Barrier System ---

barrier_db: Optional[BarrierDatabase] = None
barrier_controllers: dict[str, BarrierController] = {}


def init_barrier_controllers():
    """Initialize barrier controllers for entry/exit.

    Entry barrier uses two cameras:
      - camera5 (plate cam): approach + reading zones
      - camera6 (safety cam): safety zone
    Exit barrier: manual control only (no camera automation until camera7 is added).
    """
    global barrier_db, barrier_controllers

    barrier_db = BarrierDatabase()
    barrier_db.reset_runtime_state()
    logger.info("[Barrier] Database initialized")

    # Try to load PlateRecognizer (may fail if models not downloaded yet).
    # eager_load=True forces YOLO + parseq to download/JIT NOW so the first
    # car at the barrier doesn't pay the ~10s cold-start penalty.
    #
    # USE_TFLITE=1 switches the plate *detector* (YOLO11x) to the FP16 TFLite
    # build for edge-deployability demos. Default (unset/0) keeps the PyTorch
    # path — faster on desktop GPU and used in normal operation. OCR (parseq)
    # is unchanged in both modes.
    use_tflite = os.getenv("USE_TFLITE", "0") == "1"
    if use_tflite:
        logger.info("[Barrier] USE_TFLITE=1 — plate detector will run FP16 TFLite (edge mode)")

    plate_recognizer = None
    try:
        from plate_scanner import PlateRecognizer
        # enhance=False here because stream_frames() applies the same HSV boost
        # to the visible processed_frame BEFORE detection. Doing it once at the
        # stream layer keeps the dashboard view aligned with what the AI sees
        # and avoids double-enhancement.
        plate_recognizer = PlateRecognizer(
            confidence=0.06,
            augment=True,
            match_training_aug=True,
            enhance=False,
            eager_load=True,
            use_tflite=use_tflite,
        )
    except Exception:
        logger.exception(
            "[Barrier] PlateRecognizer not available — "
            "barrier system will run without plate recognition"
        )

    barrier_dir = Path(__file__).parent.parent / "BarrierSystem"

    # --- Entry barrier: camera5 (approach+reading) + camera6 (safety) ---
    entry_zones = {"approach_zone": None, "reading_zone": None, "safety_zone": None}

    # Load plate camera zones (camera5)
    cam5_file = barrier_dir / f"{ENTRY_PLATE_CAMERA}_barrier.p"
    if cam5_file.exists():
        try:
            with open(cam5_file, "rb") as f:
                data = pickle.load(f)
            entry_zones["approach_zone"] = data.get("approach")
            entry_zones["reading_zone"] = data.get("reading")
            logger.info("[Barrier] Entry: loaded approach+reading zones from %s", cam5_file)
        except Exception:
            logger.exception("[Barrier] Could not load %s", cam5_file)
    else:
        logger.warning(
            "[Barrier] No zone calibration for %s. "
            "Run BarrierSystem/mark_barrier_zones.py (mode: Plate camera)",
            ENTRY_PLATE_CAMERA,
        )

    # Load safety camera zones (camera6)
    cam6_file = barrier_dir / f"{ENTRY_SAFETY_CAMERA}_barrier.p"
    if cam6_file.exists():
        try:
            with open(cam6_file, "rb") as f:
                data = pickle.load(f)
            entry_zones["safety_zone"] = data.get("safety")
            logger.info("[Barrier] Entry: loaded safety zone from %s", cam6_file)
        except Exception:
            logger.exception("[Barrier] Could not load %s", cam6_file)
    else:
        logger.warning(
            "[Barrier] No zone calibration for %s. "
            "Run BarrierSystem/mark_barrier_zones.py (mode: Safety camera)",
            ENTRY_SAFETY_CAMERA,
        )

    if plate_recognizer is not None:
        entry_ctrl = BarrierController(
            barrier_id="entry",
            camera_id=ENTRY_PLATE_CAMERA,
            plate_recognizer=plate_recognizer,
            database=barrier_db,
            safety_camera_id=ENTRY_SAFETY_CAMERA,
            **entry_zones
        )
        barrier_controllers["entry"] = entry_ctrl
    else:
        logger.warning("[Barrier] Skipping entry controller (no plate recognizer)")


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

manager = ConnectionManager()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/stats")
async def get_stats():
    """Get current parking statistics for all cameras."""
    det = get_detector()
    all_stats = {}
    total_spaces = 0
    total_occupied = 0
    total_cars = 0

    for cam_id in CAMERA_IDS:
        stats = dict(det.get_stats(cam_id))
        stats["frames_received"] = frame_storages[cam_id].frame_count
        all_stats[cam_id] = stats
        total_spaces += stats.get("total_spaces", 0)
        total_occupied += stats.get("occupied", 0)
        # Only count cars from camera1 to avoid duplicates
        # (camera2 sees the same cars from a different angle)
        if cam_id == "camera1":
            total_cars += stats.get("cars_detected", 0)

    all_stats["aggregate"] = {
        "total_spaces": total_spaces,
        "occupied": total_occupied,
        "available": total_spaces - total_occupied,
        "cars_detected": total_cars,
    }
    return all_stats


@app.post("/api/frame")
async def receive_frame(request: Request):
    """
    Receive frame notification from UE5.
    UE5 sends JSON with camera_id and optionally frame_path.
    """
    try:
        data = {}
        body = await request.body()
        if body:
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                pass

        camera_id = data.get("camera_id", "camera1")

        if camera_id not in CAMERA_IDS:
            logger.warning("[FRAME] Rejected unknown camera_id=%r, data=%s", camera_id, data)
            return JSONResponse(
                {"error": f"Unknown camera_id: {camera_id}", "valid_ids": CAMERA_IDS},
                status_code=400
            )

        frame_path_str = data.get("frame_path", str(FRAME_PATHS.get(camera_id, "")))
        frame_path = Path(frame_path_str)

        if frame_path.is_dir():
            frame_path = frame_path / f"{camera_id}.png"

        if not frame_path.exists():
            return JSONResponse({
                "error": f"File not found: {frame_path}",
                "hint": "Check that Export Render Target in UE5 is working"
            }, status_code=404)

        # Retry imread up to 10 times (UE5 may be writing the file)
        frame = None
        for _ in range(10):
            try:
                frame = cv2.imread(str(frame_path))
            except PermissionError:
                time.sleep(0.05)
                continue
            if frame is not None:
                break
            time.sleep(0.05)

        if frame is None:
            # File is being written by UE5 — not an error, just skip this frame
            return {"status": "skipped", "camera_id": camera_id, "reason": "file_busy"}

        storage = frame_storages[camera_id]
        storage.current_frame = frame
        storage.last_update = time.time()
        storage.frame_count += 1

        return {"status": "ok", "camera_id": camera_id, "frame_count": storage.frame_count}

    except Exception as e:
        logger.exception("[FRAME] Error handling frame upload")
        return JSONResponse({"error": str(e)}, status_code=500)


# --- Barrier REST API ---
# NOTE: Static routes (/api/barrier/log, /api/barrier/plates) MUST be declared
# BEFORE the parametrized route /api/barrier/{barrier_id}, otherwise FastAPI
# matches the latter first with barrier_id="plates" / barrier_id="log".

@app.get("/api/barrier/log")
async def get_barrier_log():
    """Get recent access log entries for the dashboard."""
    if barrier_db is None:
        return []
    return barrier_db.get_recent_log(limit=50)


@app.get("/api/barrier/plates")
async def get_barrier_plates():
    """Get list of allowed plates."""
    if barrier_db is None:
        return []
    return barrier_db.get_all_plates()


@app.post("/api/barrier/plates")
async def add_barrier_plate(request: Request):
    """Add a plate to the allowed list."""
    if barrier_db is None:
        return JSONResponse({"error": "Barrier DB not initialized"}, status_code=500)
    data = await request.json()
    plate = data.get("plate_number", "").strip().upper()
    if not plate:
        return JSONResponse({"error": "plate_number required"}, status_code=400)
    ok = barrier_db.add_plate(plate, data.get("owner_name", ""), data.get("vehicle_description", ""))
    return {"status": "ok" if ok else "error", "plate": plate}


@app.delete("/api/barrier/plates/{plate}")
async def remove_barrier_plate(plate: str):
    """Remove a plate from the allowed list."""
    if barrier_db is None:
        return JSONResponse({"error": "Barrier DB not initialized"}, status_code=500)
    ok = barrier_db.remove_plate(plate)
    return {"status": "ok" if ok else "not_found", "plate": plate}


@app.put("/api/barrier/plates/{plate}")
async def update_barrier_plate(plate: str, request: Request):
    """Update owner/description for an existing plate."""
    if barrier_db is None:
        return JSONResponse({"error": "Barrier DB not initialized"}, status_code=500)
    data = await request.json()
    ok = barrier_db.update_plate(
        plate,
        owner=data.get("owner_name"),
        description=data.get("vehicle_description"),
    )
    return {"status": "ok" if ok else "not_found", "plate": plate}


@app.get("/api/barrier/{barrier_id}")
async def get_barrier_state(barrier_id: str):
    """UE5 polls this endpoint to get barrier state and commands."""
    controller = barrier_controllers.get(barrier_id)
    if controller is None:
        return JSONResponse({"error": f"Unknown barrier: {barrier_id}"}, status_code=404)
    return controller.get_barrier_api_state()


@app.post("/api/barrier/{barrier_id}/ue5_ack")
async def barrier_ue5_ack(barrier_id: str, request: Request):
    """UE5 calls this when barrier animation completes."""
    controller = barrier_controllers.get(barrier_id)
    if controller is None:
        return JSONResponse({"error": f"Unknown barrier: {barrier_id}"}, status_code=404)

    data = await request.json()
    event = data.get("event", "")
    controller.ack_from_ue5(event)
    return {"status": "ok", "event": event}


@app.post("/api/barrier/{barrier_id}/manual")
async def barrier_manual_control(barrier_id: str, request: Request):
    """Manual barrier control from dashboard operator."""
    controller = barrier_controllers.get(barrier_id)
    if controller is None:
        return JSONResponse({"error": f"Unknown barrier: {barrier_id}"}, status_code=404)
    data = await request.json()
    command = data.get("command", "")
    if command == "open":
        controller.manual_open()
    elif command == "close":
        controller.manual_close()
    else:
        return JSONResponse({"error": f"Unknown command: {command}"}, status_code=400)
    return {"status": "ok", "command": command}


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """WebSocket endpoint for video streaming."""
    await manager.connect(websocket)
    logger.info("[WS] Client connected. Total: %d", len(manager.active_connections))

    try:
        det = get_detector()
        streaming_task = asyncio.create_task(stream_frames(websocket, det))

        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)

                if message.get("type") == "set_mode":
                    new_mode = message.get("mode")
                    if new_mode in [ParkingDetector.MODE_PARKING_SPACES,
                                   ParkingDetector.MODE_CAR_COUNTER]:
                        det.set_mode(new_mode)
                        logger.info("[WS] Mode changed to: %s", new_mode)

                elif message.get("type") == "barrier_command":
                    bid = message.get("barrier_id")
                    cmd = message.get("command")
                    ctrl = barrier_controllers.get(bid)
                    if ctrl and cmd == "manual_open":
                        ctrl.manual_open()
                    elif ctrl and cmd == "manual_close":
                        ctrl.manual_close()

            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                pass

    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("[WS] Error in websocket handler")
    finally:
        streaming_task.cancel()
        manager.disconnect(websocket)
        logger.info("[WS] Client disconnected. Total: %d", len(manager.active_connections))


async def stream_frames(websocket: WebSocket, det: ParkingDetector):
    """Stream video frames from all cameras to a WebSocket client."""
    frame_interval = 1.0 / 10  # 10 FPS
    no_frame_counts = {cam_id: 0 for cam_id in CAMERA_IDS}

    while True:
        try:
            cameras_data = {}

            for cam_id in CAMERA_IDS:
                storage = frame_storages[cam_id]
                original_frame = None

                if storage.current_frame is not None:
                    time_since_update = time.time() - storage.last_update
                    if time_since_update < 2.0:
                        original_frame = storage.current_frame.copy()
                        no_frame_counts[cam_id] = 0

                if original_frame is not None:
                    processed_frame = original_frame.copy()

                    if cam_id == ENTRY_PLATE_CAMERA:
                        # camera5: plate recognition for entry barrier.
                        # Apply HSV brightness/saturation boost to the working
                        # frame BEFORE detection so (a) the visible "AI Detection"
                        # canvas shows the same image the AI sees and (b) the
                        # plate recognizer's internal enhance can stay off
                        # (single point of enhancement = no double-application).
                        entry_ctrl = barrier_controllers.get("entry")
                        if entry_ctrl is not None and entry_ctrl.plate_recognizer is not None:
                            processed_frame = entry_ctrl.plate_recognizer.enhance_frame(processed_frame)

                        result = det.process_frame(processed_frame, cam_id)
                        if len(result) == 3:
                            processed_frame, stats, object_list = result
                        else:
                            processed_frame, stats = result
                            object_list = []

                        if entry_ctrl is not None:
                            barrier_result = entry_ctrl.process_frame(processed_frame, object_list)
                            stats["barrier"] = barrier_result

                    elif cam_id == ENTRY_SAFETY_CAMERA:
                        # camera6: safety zone monitoring for entry barrier
                        result = det.process_frame(processed_frame, cam_id)
                        if len(result) == 3:
                            processed_frame, stats, object_list = result
                        else:
                            processed_frame, stats = result
                            object_list = []

                        entry_ctrl = barrier_controllers.get("entry")
                        if entry_ctrl is not None:
                            entry_ctrl.update_safety(object_list)
                            entry_ctrl.draw_safety_overlay(processed_frame)
                            stats["barrier"] = entry_ctrl._build_result()

                    else:
                        processed_frame, stats = det.process_frame(processed_frame, cam_id)

                    stats["ue5_connected"] = True

                    _, original_buffer = cv2.imencode('.jpg', original_frame,
                                                       [cv2.IMWRITE_JPEG_QUALITY, 70])
                    _, processed_buffer = cv2.imencode('.jpg', processed_frame,
                                                        [cv2.IMWRITE_JPEG_QUALITY, 70])

                    cameras_data[cam_id] = {
                        "original": base64.b64encode(original_buffer).decode('utf-8'),
                        "processed": base64.b64encode(processed_buffer).decode('utf-8'),
                        "stats": stats
                    }
                else:
                    no_frame_counts[cam_id] += 1

                    if no_frame_counts[cam_id] < 30:
                        msg = f"Waiting for UE5 ({cam_id})..."
                    else:
                        msg = f"No frames from UE5 ({cam_id}). Check Blueprint."

                    placeholder = create_placeholder_image(msg)
                    _, buffer = cv2.imencode('.jpg', placeholder,
                                             [cv2.IMWRITE_JPEG_QUALITY, 70])
                    placeholder_b64 = base64.b64encode(buffer).decode('utf-8')

                    cam_total = det.camera_total_spaces.get(cam_id, 12)
                    cameras_data[cam_id] = {
                        "original": placeholder_b64,
                        "processed": placeholder_b64,
                        "stats": {
                            "total_spaces": cam_total,
                            "occupied": 0,
                            "available": cam_total,
                            "cars_detected": 0,
                            "mode": det.mode,
                            "ue5_connected": False
                        }
                    }

            await websocket.send_json({
                "type": "frame",
                "cameras": cameras_data
            })

            await asyncio.sleep(frame_interval)

        except Exception:
            logger.exception("[WS] Streaming error")
            await asyncio.sleep(1)
            break


def main():
    """Run the server."""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Smart Parking Monitor server")
    parser.add_argument(
        "--tflite",
        action="store_true",
        help="Run plate detector as FP16 TFLite (edge demo). Default: PyTorch.",
    )
    args = parser.parse_args()

    # Surface --tflite to init_barrier_controllers() via env (it runs inside
    # FastAPI's lifespan, after uvicorn forks off this entrypoint).
    if args.tflite:
        os.environ["USE_TFLITE"] = "1"

    print_banner("Smart Parking Monitor · Multi-Camera + Barrier Control · v2.0.0")
    print_section("Configuration")
    logger.info("Frames directory: %s", FRAMES_DIR)
    logger.info("Cameras: %s", ", ".join(CAMERA_IDS))
    for cam_id, path in FRAME_PATHS.items():
        logger.info("  ▸ %s  %s", cam_id, path)
    logger.info("Dashboard: http://localhost:8000  (Ctrl+C to stop)")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
