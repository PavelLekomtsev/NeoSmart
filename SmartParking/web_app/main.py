"""
Smart Parking Web Application
FastAPI server - receives frames from UE5 via HTTP.
"""

import asyncio
import base64
import json
import cv2
import numpy as np
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from detector import ParkingDetector, create_placeholder_image

app = FastAPI(
    title="Smart Parking Monitor",
    description="Real-time parking monitoring with AI detection",
    version="1.0.0"
)

BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

FRAME_PATH = Path("E:/Work/Computer_Vision/Projects/NeoSmart/SmartParking/frames/frame.png")


class FrameStorage:
    def __init__(self):
        self.current_frame: Optional[np.ndarray] = None
        self.last_update: float = 0
        self.frame_count: int = 0

frame_storage = FrameStorage()

detector = None

def get_detector() -> ParkingDetector:
    """Get or create the detector instance."""
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
    """Get current parking statistics."""
    det = get_detector()
    stats = det.get_stats()
    stats["frames_received"] = frame_storage.frame_count
    return stats


@app.post("/api/frame")
async def receive_frame(request: Request):
    """
    Receive frame notification from UE5.
    UE5 sends JSON with frame_path, Python reads the file.
    """
    try:
        data = await request.json()

        frame_path_str = data.get("frame_path", str(FRAME_PATH))
        camera_id = data.get("camera_id", "unknown")

        frame_path = Path(frame_path_str)

        # Auto-fix: if path is a directory, append frame.png
        if frame_path.is_dir():
            frame_path = frame_path / "frame.png"
            print(f"[{camera_id}] Path was a directory, fixed to: {frame_path}")

        print(f"[{camera_id}] Checking: {frame_path}")

        if not frame_path.exists():
            print(f"[{camera_id}] ERROR: File does not exist: {frame_path}")
            return JSONResponse({
                "error": f"File not found: {frame_path}",
                "hint": "Check that Export Render Target in UE5 is working"
            }, status_code=404)

        frame = cv2.imread(str(frame_path))

        if frame is None:
            print(f"[{camera_id}] ERROR: Failed to decode image: {frame_path}")
            return JSONResponse({
                "error": "Failed to decode image file",
                "path": str(frame_path)
            }, status_code=400)

        frame_storage.current_frame = frame
        frame_storage.last_update = time.time()
        frame_storage.frame_count += 1

        print(f"[{camera_id}] Frame #{frame_storage.frame_count} received ({frame.shape[1]}x{frame.shape[0]})")

        return {"status": "ok", "frame_count": frame_storage.frame_count}

    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON: {e}")
        return JSONResponse({"error": f"Invalid JSON: {e}"}, status_code=400)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """WebSocket endpoint for video streaming."""
    await manager.connect(websocket)
    det = get_detector()

    print(f"Client connected. Total: {len(manager.active_connections)}")

    try:
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
                        print(f"Mode changed to: {new_mode}")

            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                pass

    except WebSocketDisconnect:
        pass
    finally:
        streaming_task.cancel()
        manager.disconnect(websocket)
        print(f"Client disconnected. Total: {len(manager.active_connections)}")


async def stream_frames(websocket: WebSocket, det: ParkingDetector):
    """Stream video frames to a WebSocket client."""
    frame_interval = 1.0 / 15  # 15 FPS
    no_frame_count = 0

    while True:
        try:
            original_frame = None

            if frame_storage.current_frame is not None:
                time_since_update = time.time() - frame_storage.last_update
                if time_since_update < 2.0:
                    original_frame = frame_storage.current_frame.copy()
                    no_frame_count = 0

            if original_frame is not None:
                processed_frame = original_frame.copy()
                processed_frame, stats = det.process_frame(processed_frame)
                stats["ue5_connected"] = True

                _, original_buffer = cv2.imencode('.jpg', original_frame,
                                                   [cv2.IMWRITE_JPEG_QUALITY, 70])
                _, processed_buffer = cv2.imencode('.jpg', processed_frame,
                                                    [cv2.IMWRITE_JPEG_QUALITY, 70])

                original_b64 = base64.b64encode(original_buffer).decode('utf-8')
                processed_b64 = base64.b64encode(processed_buffer).decode('utf-8')

                await websocket.send_json({
                    "type": "frame",
                    "original": original_b64,
                    "processed": processed_b64,
                    "stats": stats
                })
            else:
                no_frame_count += 1

                if no_frame_count < 30:
                    msg = "Waiting for UE5..."
                else:
                    msg = "No frames from UE5. Check Blueprint."

                placeholder = create_placeholder_image(msg)
                _, buffer = cv2.imencode('.jpg', placeholder,
                                         [cv2.IMWRITE_JPEG_QUALITY, 70])
                placeholder_b64 = base64.b64encode(buffer).decode('utf-8')

                await websocket.send_json({
                    "type": "frame",
                    "original": placeholder_b64,
                    "processed": placeholder_b64,
                    "stats": {
                        "total_spaces": det.total_spaces,
                        "occupied": 0,
                        "available": det.total_spaces,
                        "cars_detected": 0,
                        "mode": det.mode,
                        "ue5_connected": False
                    }
                })

            await asyncio.sleep(frame_interval)

        except Exception as e:
            print(f"Streaming error: {e}")
            await asyncio.sleep(1)
            break


def main():
    """Run the server."""
    import uvicorn

    print("=" * 60)
    print("  Smart Parking Monitor")
    print("=" * 60)
    print()
    print(f"  Frame path: {FRAME_PATH}")
    print(f"  Folder exists: {FRAME_PATH.parent.exists()}")
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
