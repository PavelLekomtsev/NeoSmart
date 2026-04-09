"""
Smart Parking Web Application
FastAPI server - receives frames from UE5 via HTTP.
Supports multiple cameras.
"""

import asyncio
import base64
import json
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

from detector import ParkingDetector, create_placeholder_image


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Pre-load the detector at server startup so WebSocket connects instantly."""
    print("Pre-loading detector at startup...")
    get_detector()
    print("Server ready for connections.")
    yield


app = FastAPI(
    title="Smart Parking Monitor",
    description="Real-time parking monitoring with AI detection",
    version="2.0.0",
    lifespan=lifespan,
)

BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

FRAMES_DIR = Path("E:/Work/Computer_Vision/Projects/NeoSmart/SmartParking/frames")

CAMERA_IDS = ["camera1", "camera2", "camera3"]

FRAME_PATHS = {
    "camera1": FRAMES_DIR / "camera1.png",
    "camera2": FRAMES_DIR / "camera2.png",
    "camera3": FRAMES_DIR / "camera3.png",
}


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
        print("Initializing parking detector...")
        detector = ParkingDetector()
        print("Detector ready!")
    return detector


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
            print(f"[FRAME] Rejected unknown camera_id='{camera_id}', data={data}")
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

        # Retry imread up to 5 times (UE5 may be writing the file)
        frame = None
        for _ in range(5):
            try:
                frame = cv2.imread(str(frame_path))
            except PermissionError:
                time.sleep(0.03)
                continue
            if frame is not None:
                break
            time.sleep(0.03)

        if frame is None:
            print(f"[FRAME] Failed to decode: camera_id='{camera_id}', path='{frame_path}', exists={frame_path.exists()}")
            return JSONResponse({
                "error": "Failed to decode image file",
                "path": str(frame_path)
            }, status_code=400)

        storage = frame_storages[camera_id]
        storage.current_frame = frame
        storage.last_update = time.time()
        storage.frame_count += 1

        return {"status": "ok", "camera_id": camera_id, "frame_count": storage.frame_count}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """WebSocket endpoint for video streaming."""
    await manager.connect(websocket)
    print(f"[WS] Client connected. Total: {len(manager.active_connections)}")

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
                        print(f"[WS] Mode changed to: {new_mode}")

            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                pass

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS] Error in websocket handler: {e}")
        import traceback
        traceback.print_exc()
    finally:
        streaming_task.cancel()
        manager.disconnect(websocket)
        print(f"[WS] Client disconnected. Total: {len(manager.active_connections)}")


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

        except Exception as e:
            print(f"[WS] Streaming error: {e}")
            await asyncio.sleep(1)
            break


def main():
    """Run the server."""
    import uvicorn

    print("=" * 60)
    print("  Smart Parking Monitor (Multi-Camera)")
    print("=" * 60)
    print()
    print(f"  Frames directory: {FRAMES_DIR}")
    print(f"  Cameras: {', '.join(CAMERA_IDS)}")
    for cam_id, path in FRAME_PATHS.items():
        print(f"    {cam_id}: {path}")
    print()
    print("  Open http://localhost:8000 in your browser")
    print()
    print("  Press Ctrl+C to stop")
    print("=" * 60)

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
