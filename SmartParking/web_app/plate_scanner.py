"""
PlateScanner Integration Module
Wraps PlateScanner (git submodule) for license plate detection and OCR.
Uses YOLO for plate detection and ParseQ for text recognition.
"""

import os
import sys
import time
import warnings
import cv2
import numpy as np
from pathlib import Path
from collections import Counter

# parseq (via torch.hub) still imports from timm.models.helpers which triggers
# a FutureWarning every call. It's benign but clutters startup logs.
warnings.filterwarnings(
    "ignore",
    message=r".*Importing from timm\.models\.helpers is deprecated.*",
    category=FutureWarning,
)

# Setup PlateScanner environment before importing
_PLATE_SCANNER_DIR = Path(__file__).parent.parent / "PlateScanner"
os.environ.setdefault("PLATESCANNER_ROOT_PATH", str(_PLATE_SCANNER_DIR))

if str(_PLATE_SCANNER_DIR) not in sys.path:
    sys.path.insert(0, str(_PLATE_SCANNER_DIR))

from ultralytics import YOLO


class PlateRecognizer:
    """
    License plate detection and recognition using PlateScanner models.
    Two-stage pipeline: YOLO plate detection -> ParseQ OCR.
    """

    def __init__(self, plate_model_path: str = None,
                 confidence: float = 0.06,
                 augment: bool = True,
                 match_training_aug: bool = True,
                 enhance: bool = True,
                 brightness: float = 1.35,
                 saturation: float = 1.4,
                 eager_load: bool = True):
        """
        Initialize plate recognizer.

        Args:
            plate_model_path: Path to YOLO plate detection model weights.
                              If None, looks in Models/plate_scanner/
            confidence: YOLO confidence threshold. PlateScanner's own default is
                        0.06; we mirror it because the model is well-calibrated.
            augment: If True, pass augment=True to YOLO (test-time augmentation).
                     Matches the upstream PlateScanner predict mode and improves
                     recall at the cost of speed.
            match_training_aug: If True, apply ToGray + CLAHE to the input frame
                                before detection to mimic the training pipeline.
            enhance: If True, boost brightness + saturation on the input frame
                     before detection/OCR. Helps on dim, washed-out frames
                     (white plate + black letters lose contrast in low light).
            brightness: HSV V multiplier for `enhance` (1.0 = no change).
            saturation: HSV S multiplier for `enhance` (1.0 = no change).
            eager_load: If True, load OCR + warm up YOLO/parseq in __init__.
                        Avoids paying the ~10s first-call cost when a real car
                        arrives. Set False for tests where you only need detection.
        """
        base_dir = Path(__file__).parent.parent.parent  # NeoSmart root

        if plate_model_path is None:
            model_dir = base_dir / "Models" / "plate_scanner"
            # Try to find any .pt model in the directory
            candidates = list(model_dir.glob("*.pt")) if model_dir.exists() else []
            if candidates:
                plate_model_path = str(candidates[0])
                print(f"[PlateScanner] Using plate model: {candidates[0].name}")
            else:
                # Fallback to PlateScanner models directory
                ps_models = _PLATE_SCANNER_DIR / "models"
                candidates = list(ps_models.glob("*.pt")) if ps_models.exists() else []
                if candidates:
                    plate_model_path = str(candidates[0])
                    print(f"[PlateScanner] Using model from submodule: {candidates[0].name}")
                else:
                    raise FileNotFoundError(
                        f"No plate detection model found. "
                        f"Download from PlateScanner Yandex Disk and place in {model_dir}/"
                    )

        self.plate_detector = YOLO(plate_model_path)
        self.plate_confidence = confidence
        self.augment = augment
        self.match_training_aug = match_training_aug
        self.enhance = enhance
        self.brightness = brightness
        self.saturation = saturation

        self._rec_model = None
        self._preprocess_fn = None
        self._train_aug_pipeline = None

        print(f"[PlateScanner] Plate detector loaded: {plate_model_path}")

        if eager_load:
            self._ensure_ocr_loaded()
            self._warmup()

    def enhance_frame(self, img_bgr: np.ndarray) -> np.ndarray:
        """Boost brightness + saturation in HSV space.
        Public utility — callers can apply this explicitly to the visible
        display frame so the dashboard shows the same boosted image the
        AI pipeline sees. Dim frames with white plates + black letters wash
        out — boosting V and S widens dynamic range so YOLO and parseq see
        crisper contrast."""
        if self.brightness == 1.0 and self.saturation == 1.0:
            return img_bgr
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * self.saturation, 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * self.brightness, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Back-compat alias for existing test script
    _enhance_frame = enhance_frame

    def _warmup(self):
        """Run a single dummy inference end-to-end so torch.hub finishes
        downloading parseq + YOLO graphs are JIT-compiled before any real frame."""
        try:
            from PIL import Image as _Img
            dummy_bgr = np.zeros((640, 640, 3), dtype=np.uint8)
            self.plate_detector(dummy_bgr, verbose=False, conf=self.plate_confidence)
            if self._rec_model is not None and self._preprocess_fn is not None:
                dummy_pil = _Img.new("L", (128, 32))
                preprocessed = self._preprocess_fn(dummy_pil)
                self._rec_model("parseq", preprocessed)
            print("[PlateScanner] Warmup complete (YOLO + parseq ready)")
        except Exception as e:
            print(f"[PlateScanner] Warmup failed (will retry on first call): {e}")

    def _apply_training_aug(self, img_bgr: np.ndarray) -> np.ndarray:
        """Mimic the PlateScanner training pipeline: ToGray + CLAHE.
        Implemented with cv2 to avoid an albumentations dependency."""
        if self._train_aug_pipeline is None:
            self._train_aug_pipeline = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        equalized = self._train_aug_pipeline.apply(gray)
        return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

    def _ensure_ocr_loaded(self):
        """Lazy-load the ParseQ OCR model on first use.

        Uses our own port of PlateScanner's `preprocess_license_plate` +
        `RecognitionModel` so we don't pull in the broken `platescanner.utils`
        package init chain (which transitively imports `cvtk`)."""
        if self._rec_model is not None:
            return

        try:
            self._rec_model = _LocalRecognitionModel()
            self._preprocess_fn = _preprocess_license_plate
            print("[PlateScanner] ParseQ OCR model loaded (local port)")
        except Exception as e:
            print(f"[PlateScanner] Warning: Could not load ParseQ OCR: {e}")
            print("[PlateScanner] Falling back to detection-only mode")
            self._rec_model = None
            self._preprocess_fn = None

    def detect_plate(self, frame: np.ndarray, car_bbox: tuple = None) -> dict | None:
        """
        Detect and read a license plate from a frame.

        Args:
            frame: Full frame in BGR format (OpenCV)
            car_bbox: Optional (x1, y1, x2, y2) to crop search area to car region

        Returns:
            dict with keys: plate_text, confidence, bbox (x1,y1,x2,y2 in frame coords),
            plate_image (cropped plate as numpy array), or None if no plate found
        """
        # Apply enhancement to the full frame first so both detection and the
        # OCR crop work on the boosted version.
        if self.enhance:
            frame = self.enhance_frame(frame)

        # Optionally crop to car region for faster/more accurate detection
        search_region = frame
        offset_x, offset_y = 0, 0

        if car_bbox is not None:
            x1, y1, x2, y2 = car_bbox
            # Expand slightly for plates that extend beyond car bbox
            h, w = frame.shape[:2]
            pad_x = int((x2 - x1) * 0.05)
            pad_y = int((y2 - y1) * 0.1)
            cx1 = max(0, x1 - pad_x)
            cy1 = max(0, y1 - pad_y)
            cx2 = min(w, x2 + pad_x)
            cy2 = min(h, y2 + pad_y)
            search_region = frame[cy1:cy2, cx1:cx2]
            offset_x, offset_y = cx1, cy1

        if search_region.size == 0:
            return None

        detect_input = search_region
        if self.match_training_aug:
            detect_input = self._apply_training_aug(detect_input)

        # Run YOLO plate detection
        results = self.plate_detector(detect_input, stream=False, verbose=False,
                                       conf=self.plate_confidence,
                                       augment=self.augment)

        best_plate = None
        best_conf = 0.0

        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf > best_conf:
                    px1, py1, px2, py2 = box.xyxy[0].cpu().numpy()
                    px1, py1, px2, py2 = int(px1), int(py1), int(px2), int(py2)

                    # Convert back to full frame coordinates
                    abs_x1 = px1 + offset_x
                    abs_y1 = py1 + offset_y
                    abs_x2 = px2 + offset_x
                    abs_y2 = py2 + offset_y

                    best_plate = {
                        "bbox": (abs_x1, abs_y1, abs_x2, abs_y2),
                        "confidence": conf,
                        "local_bbox": (px1, py1, px2, py2),
                    }
                    best_conf = conf

        if best_plate is None:
            return None

        # Crop plate image
        lx1, ly1, lx2, ly2 = best_plate["local_bbox"]
        plate_crop = search_region[ly1:ly2, lx1:lx2]

        if plate_crop.size == 0:
            return None

        # Run OCR
        plate_text = ""
        self._ensure_ocr_loaded()

        if self._rec_model is not None and self._preprocess_fn is not None:
            try:
                from PIL import Image
                # Convert BGR to RGB, then to PIL
                plate_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
                plate_pil = Image.fromarray(plate_rgb)
                preprocessed = self._preprocess_fn(plate_pil)
                plate_text, _ = self._rec_model("parseq", preprocessed)
            except Exception as e:
                print(f"[PlateScanner] OCR error: {e}")
                plate_text = ""

        return {
            "plate_text": plate_text,
            "confidence": best_conf,
            "bbox": best_plate["bbox"],
            "plate_image": plate_crop,
        }

    def detect_plate_majority_vote(self, frames: list[np.ndarray],
                                    car_bboxes: list[tuple] = None,
                                    min_votes: int = 2) -> dict | None:
        """
        Run plate detection across multiple frames and return majority-vote result.
        This improves OCR reliability by accumulating readings.

        Args:
            frames: List of frames (BGR numpy arrays)
            car_bboxes: Optional list of (x1,y1,x2,y2) per frame
            min_votes: Minimum times same plate must be read to be accepted

        Returns:
            dict with plate_text, confidence, bbox, plate_image of best reading,
            or None if no consistent plate found
        """
        readings = []

        for i, frame in enumerate(frames):
            car_bbox = car_bboxes[i] if car_bboxes and i < len(car_bboxes) else None
            result = self.detect_plate(frame, car_bbox)
            if result and result["plate_text"]:
                readings.append(result)

        if not readings:
            return None

        # Majority vote on plate text
        plate_texts = [r["plate_text"] for r in readings]
        counter = Counter(plate_texts)
        most_common_text, count = counter.most_common(1)[0]

        if count < min_votes:
            # Not enough consistent readings — return best single reading
            best = max(readings, key=lambda r: r["confidence"])
            return best

        # Return the highest-confidence reading with the majority text
        matching = [r for r in readings if r["plate_text"] == most_common_text]
        best = max(matching, key=lambda r: r["confidence"])
        return best


# ----------------------------------------------------------------------------
# Local port of PlateScanner's OCR pipeline (platescanner/utils/plates.py)
#
# Vendored to avoid importing `platescanner.utils`, whose package init
# transitively depends on `cvtk` (a separate, not-installed package).
# Keeps the same behaviour: gray + CLAHE(clip=2, tile=1) + erode(1x1)
# preprocessing, parseq via torch.hub, and the same post-processing rules
# tuned for Russian-style plates.
# ----------------------------------------------------------------------------
import re as _re
from PIL import Image as _PILImage


def _preprocess_license_plate(plate_image_pil):
    """OCR-side plate preprocessing.

    Based on upstream `platescanner.utils.plates.preprocess_license_plate`
    (gray + CLAHE clip=2 tile=(1,1) + 1x1 erosion). We push contrast harder
    here than upstream because the parseq model is a generic scene-text
    network and benefits from a wider dynamic range — black letters become
    blacker and the white plate background becomes whiter:

      1) ToGray
      2) CLAHE (clip=4, tile=(8,8)) — local adaptive contrast
      3) NORM_MINMAX stretch — push the per-plate min/max to the full [0,255]
      4) Light unsharp mask — sharpens character edges
    """
    plate_np = np.array(plate_image_pil)
    if plate_np.ndim == 3 and plate_np.shape[-1] >= 3:
        plate_np = cv2.cvtColor(plate_np, cv2.COLOR_RGB2GRAY)
    elif plate_np.ndim == 3 and plate_np.shape[-1] == 1:
        plate_np = plate_np[..., 0]

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    plate_np = clahe.apply(plate_np)

    plate_np = cv2.normalize(plate_np, None, 0, 255, cv2.NORM_MINMAX)

    blurred = cv2.GaussianBlur(plate_np, (0, 0), sigmaX=1.0)
    plate_np = cv2.addWeighted(plate_np, 1.5, blurred, -0.5, 0)

    return _PILImage.fromarray(plate_np)


class _LocalRecognitionModel:
    """Replicates platescanner.utils.plates.RecognitionModel using torch.hub."""

    models = ["parseq", "parseq_tiny", "abinet", "crnn", "trba", "vitstr"]

    def __init__(self):
        import torch
        import torchvision.transforms as T

        self._torch = torch
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model_cache = {}
        self._preprocess = T.Compose([
            T.Resize((32, 128), T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(0.5, 0.5),
        ])

    def _get_model(self, name: str):
        if name in self._model_cache:
            return self._model_cache[name]
        model = self._torch.hub.load("baudm/parseq", name, pretrained=True).eval().to(self.device)
        self._model_cache[name] = model
        return model

    def __call__(self, model_name: str, image) -> tuple[str, list]:
        torch = self._torch
        with torch.inference_mode():
            model = self._get_model(model_name)
            tensor = self._preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
            pred = model(tensor).softmax(-1)
            label, _ = model.tokenizer.decode(pred)
            raw_label, raw_confidence = model.tokenizer.decode(pred, raw=True)
            max_len = 25 if model_name == "crnn" else len(label[0]) + 1
            conf = list(map("{:0.1f}".format, raw_confidence[0][:max_len].tolist()))
            text = label[0]
            if text:
                text = self.process_text(text)
            return text, [raw_label[0][:max_len], conf]

    # Russian private-car plate format (9 chars): L D D D L L D D D
    #   L = letter from the Russian-Latin overlap set: A B E K M H O P C T Y X
    #   D = digit 0..9
    # Positions are 0-indexed.
    _LETTER_POS = (0, 4, 5)
    _DIGIT_POS = (1, 2, 3, 6, 7, 8)
    _RUSSIAN_PLATE_LETTERS = set("ABEKMHOPCTYX")

    # Visual digit -> letter (used in letter positions only)
    _DIGIT_TO_LETTER = {
        "0": "O", "1": "I", "2": "Z", "3": "E",
        "4": "A", "5": "S", "6": "G", "7": "T",
        "8": "B", "9": "G",
    }
    # Visual letter -> digit (used in digit positions only)
    _LETTER_TO_DIGIT = {
        "O": "0", "Q": "0", "D": "0",
        "I": "1", "L": "1", "J": "1",
        "Z": "2",
        "E": "3",
        "A": "4",
        "S": "5",
        "G": "6",
        "T": "7",
        "B": "8",
    }
    # Latin letters NOT in Russian-Latin overlap, mapped to nearest visual valid letter
    # (used in letter positions only, after digit/letter pass).
    _NON_RUSSIAN_TO_RUSSIAN = {
        "F": "E", "G": "C", "Q": "O", "R": "B",
        "V": "Y", "U": "Y", "W": "X", "N": "H",
        "D": "O", "I": "T", "J": "T", "L": "T",
        "S": "B", "Z": "B",
    }

    # Full-plate validator. Accepts 8-char (2-digit region) and 9-char
    # (3-digit region) Russian private-car plates. Anything shorter or with
    # characters in the wrong position is a truncated/garbage OCR read and
    # must be dropped before it ever reaches the whitelist check.
    _PLATE_REGEX = _re.compile(r"^[ABEKMHOPCTYX]\d{3}[ABEKMHOPCTYX]{2}\d{2,3}$")

    @classmethod
    def process_text(cls, recognized_text: str) -> str:
        """Normalize OCR output to Russian private-car plate format.

        Format (9 chars): LDDDLL + DDD where:
          - position 0:  letter (A,B,E,K,M,H,O,P,C,T,Y,X)
          - positions 1-3: digits (000-999)
          - positions 4-5: letters
          - positions 6-8: digits (3-digit region code)

        Pipeline:
          1) Strip non-alphanumeric, uppercase.
          2) Per-position digit<->letter swap based on visual similarity
             (5<->S, 0<->O, 8<->B, 1<->I, etc.).
          3) In letter positions, map any Latin letter outside the Russian-plate
             set to the nearest visually valid letter (F->E, R->B, N->H, ...).
          4) Truncate to 9 chars.
        """
        text = _re.sub(r"[^A-Za-z0-9]", "", recognized_text).upper()
        if not text:
            return ""

        chars = list(text)

        # Pass 1: position-aware digit<->letter normalization
        for i, c in enumerate(chars):
            if i in cls._LETTER_POS and c.isdigit():
                chars[i] = cls._DIGIT_TO_LETTER.get(c, c)
            elif i in cls._DIGIT_POS and c.isalpha():
                chars[i] = cls._LETTER_TO_DIGIT.get(c, c)

        # Pass 2: in letter positions, force Russian-plate letter set
        for i in cls._LETTER_POS:
            if i >= len(chars):
                continue
            c = chars[i]
            if c.isalpha() and c not in cls._RUSSIAN_PLATE_LETTERS:
                chars[i] = cls._NON_RUSSIAN_TO_RUSSIAN.get(c, c)

        candidate = "".join(chars)[:9]

        # Reject anything that doesn't match a full 8/9-char plate. Partial reads
        # like "M214OA" (letter cut off) would otherwise look plausible, poison
        # the OCR consensus buffer, and in the worst case get logged as DENIED
        # for a car whose real plate is whitelisted. Dropping them forces the
        # controller to wait for a better frame.
        if not cls._PLATE_REGEX.match(candidate):
            return ""
        return candidate
