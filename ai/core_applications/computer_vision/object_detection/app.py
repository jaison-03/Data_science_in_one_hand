"""YOLOv8 object detection with optional Groq-powered scene analysis."""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from groq import Groq, GroqError
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[4]
load_dotenv(ROOT / ".env", override=False)
GROQ_MODEL = os.getenv("GROQ_MODEL", "").strip() or "openai/gpt-oss-20b"
MODEL_URL = (
    "https://github.com/CVHub520/X-AnyLabeling/releases/download/"
    "v0.1.0/yolov8n.onnx"
)
MODEL_SHA1 = "68f864475d06e2ec4037181052739f268eeac38d"
MODEL_PATH = Path.home() / ".cache" / "my_ai_project" / "yolov8n.onnx"
MODEL_INPUT_SIZE = 640
MAX_IMAGE_PIXELS = 24_000_000
MAX_MODEL_BYTES = 60_000_000
SAMPLES = {
    "People and bus": "https://ultralytics.com/images/bus.jpg",
    "People playing soccer": "https://ultralytics.com/images/zidane.jpg",
    "Cats at home": (
        "https://images.cocodataset.org/val2017/000000039769.jpg"
    ),
}
COCO_LABELS = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase",     "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
)


if not st.session_state.get("_embedded_ai_project"):
    st.set_page_config(
        page_title="Object Detection Studio",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded",
    )

st.markdown(
    """
    <style>
    :root {
        --od-accent: #49dcb1;
        --od-accent-soft: rgba(73, 220, 177, .12);
        --od-card: color-mix(in srgb, var(--secondary-background-color) 92%, #49dcb1 8%);
        --od-border: color-mix(in srgb, var(--text-color) 14%, transparent);
        --od-muted: color-mix(in srgb, var(--text-color) 68%, transparent);
    }
    .stApp { background:
        radial-gradient(ellipse at 80% 0%, rgba(73,220,177,.09), transparent 36%),
        var(--background-color); }
    .block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem; }
    .od-hero {
        border: 1px solid var(--od-border); border-radius: 24px;
        padding: 1.7rem 2rem; margin: .5rem 0 1.4rem;
        background: linear-gradient(120deg, var(--od-card), var(--od-accent-soft));
    }
    .od-eyebrow { color: var(--od-accent); font-size: .75rem; font-weight: 750;
        letter-spacing: .15em; text-transform: uppercase; }
    .od-hero h1 { margin: .25rem 0 .35rem; letter-spacing: -.04em; }
    .od-hero p, .od-muted { color: var(--od-muted); }
    .od-metric {
        min-height: 112px; padding: 1rem 1.1rem; border-radius: 18px;
        border: 1px solid var(--od-border); background: var(--od-card);
    }
    .od-metric-label { color: var(--od-muted); font-size: .82rem; }
    .od-metric-value { margin-top: .45rem; font-size: 1.7rem;
        font-weight: 750; letter-spacing: -.03em; }
    .od-section {
        padding: 1rem 1.15rem; margin: .6rem 0;
        border: 1px solid var(--od-border); border-radius: 16px;
        background: var(--od-card);
    }
    div[data-testid="stFileUploader"] section { border-radius: 14px; }
    .stButton > button[kind="primary"] { border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="od-hero">
      <div class="od-eyebrow">Computer vision · real-time inference</div>
      <h1>Object Detection Studio</h1>
      <p>Detect and inspect 80 everyday object classes with YOLOv8 Nano.
      Add an optional Groq-powered scene brief after the visual analysis.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Preparing the YOLOv8 Nano detector…")
def load_detector() -> dict:
    """Download once and initialize the small pretrained ONNX detector."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.is_file() or MODEL_PATH.stat().st_size == 0:
        temporary_path: Path | None = None
        try:
            request = urllib.request.Request(
                MODEL_URL, headers={"User-Agent": "ObjectDetectionStudio/1.0"}
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                declared_size = response.headers.get("Content-Length")
                if declared_size and int(declared_size) > MAX_MODEL_BYTES:
                    raise ValueError("The detector model download exceeds 60 MB.")
                with tempfile.NamedTemporaryFile(
                    mode="wb", dir=MODEL_PATH.parent, suffix=".download",
                    delete=False,
                ) as temporary:
                    temporary_path = Path(temporary.name)
                    total = 0
                    while chunk := response.read(1024 * 1024):
                        total += len(chunk)
                        if total > MAX_MODEL_BYTES:
                            raise ValueError(
                                "The detector model download exceeds 60 MB."
                            )
                        temporary.write(chunk)
            if temporary_path.stat().st_size < 1_000_000:
                raise ValueError("The downloaded detector model is incomplete.")
            digest = hashlib.sha1(usedforsecurity=False)
            with temporary_path.open("rb") as downloaded_model:
                while chunk := downloaded_model.read(1024 * 1024):
                    digest.update(chunk)
            if digest.hexdigest() != MODEL_SHA1:
                raise ValueError("The detector model failed its integrity check.")
            temporary_path.replace(MODEL_PATH)
        except (OSError, urllib.error.URLError, ValueError):
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise

    network = cv2.dnn.readNetFromONNX(str(MODEL_PATH))
    network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return {"network": network, "lock": Lock()}


@st.cache_data(show_spinner=False, max_entries=6)
def fetch_sample_image(name: str) -> bytes:
    """Fetch and cache a public example photograph for the sample gallery."""
    url = SAMPLES[name]
    request = urllib.request.Request(
        url, headers={"User-Agent": "ObjectDetectionStudio/1.0"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = response.read(MAX_IMAGE_PIXELS * 2)
    if not payload:
        raise ValueError("The sample image download was empty.")
    return payload


def open_image(payload: bytes) -> Image.Image:
    """Decode, orient, and bound image dimensions before processing."""
    try:
        with Image.open(io.BytesIO(payload)) as source:
            image = ImageOps.exif_transpose(source)
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ValueError("Images must be no larger than 24 megapixels.")
            return image.convert("RGB")
    except Image.DecompressionBombError as error:
        raise ValueError("Images must be no larger than 24 megapixels.") from error


def letterbox(image: Image.Image) -> tuple[np.ndarray, float, int, int]:
    """Resize into the model's square input while preserving image geometry."""
    image_array = np.asarray(image)
    height, width = image_array.shape[:2]
    scale = min(MODEL_INPUT_SIZE / width, MODEL_INPUT_SIZE / height)
    resized_width = round(width * scale)
    resized_height = round(height * scale)
    resized = cv2.resize(
        image_array, (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    pad_x = (MODEL_INPUT_SIZE - resized_width) // 2
    pad_y = (MODEL_INPUT_SIZE - resized_height) // 2
    canvas = np.full(
        (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3), 114, dtype=np.uint8
    )
    canvas[pad_y:pad_y + resized_height, pad_x:pad_x + resized_width] = resized
    return canvas, scale, pad_x, pad_y


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Compute intersection-over-union between one box and a box array."""
    top_left = np.maximum(box[:2], boxes[:, :2])
    bottom_right = np.minimum(box[2:], boxes[:, 2:])
    intersection_size = np.maximum(bottom_right - top_left, 0)
    intersection = intersection_size[:, 0] * intersection_size[:, 1]
    box_area = np.prod(np.maximum(box[2:] - box[:2], 0))
    areas = np.prod(np.maximum(boxes[:, 2:] - boxes[:, :2], 0), axis=1)
    return intersection / np.maximum(box_area + areas - intersection, 1e-8)


def detect_objects(
    image: Image.Image, confidence_threshold: float, nms_threshold: float
) -> tuple[list[dict], float]:
    """Run YOLOv8 ONNX inference and class-aware non-maximum suppression."""
    detector = load_detector()
    canvas, scale, pad_x, pad_y = letterbox(image)
    blob = cv2.dnn.blobFromImage(
        canvas, scalefactor=1 / 255, size=(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        swapRB=True, crop=False,
    )
    started = time.perf_counter()
    with detector["lock"]:
        detector["network"].setInput(blob)
        outputs = detector["network"].forward()
    elapsed_ms = (time.perf_counter() - started) * 1000

    predictions = np.asarray(outputs).squeeze()
    if predictions.ndim != 2:
        raise RuntimeError(
            f"Unexpected detector output dimensions: {predictions.shape}."
        )
    if predictions.shape[0] == len(COCO_LABELS) + 4:
        predictions = predictions.T
    if predictions.shape[1] != len(COCO_LABELS) + 4:
        raise RuntimeError(
            f"Unexpected detector output shape: {predictions.shape}."
        )

    class_scores = predictions[:, 4:]
    class_ids = class_scores.argmax(axis=1)
    scores = class_scores[np.arange(len(class_ids)), class_ids]
    keep = scores >= confidence_threshold
    candidates = predictions[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]
    if not len(candidates):
        return [], elapsed_ms

    centers = candidates[:, :2]
    sizes = candidates[:, 2:4]
    boxes = np.column_stack((centers - sizes / 2, centers + sizes / 2))
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, image.width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, image.height)

    selected: list[int] = []
    for class_id in np.unique(class_ids):
        class_indices = np.flatnonzero(class_ids == class_id)
        class_indices = class_indices[np.argsort(scores[class_indices])[::-1]]
        while len(class_indices):
            best = int(class_indices[0])
            selected.append(best)
            if len(class_indices) == 1:
                break
            overlaps = box_iou(boxes[best], boxes[class_indices[1:]])
            class_indices = class_indices[1:][overlaps <= nms_threshold]

    results = []
    for index in sorted(selected, key=lambda item: scores[item], reverse=True):
        x_min, y_min, x_max, y_max = boxes[index]
        results.append({
            "class": COCO_LABELS[int(class_ids[index])],
            "confidence": float(scores[index]),
            "xmin": int(round(x_min)),
            "ymin": int(round(y_min)),
            "xmax": int(round(x_max)),
            "ymax": int(round(y_max)),
        })
    return results, elapsed_ms


def class_color(class_name: str) -> tuple[int, int, int]:
    """Generate a stable, vivid annotation color for each class."""
    digest = hashlib.sha256(class_name.encode("utf-8")).digest()
    hue = int.from_bytes(digest[:2], "big") / 65535
    import colorsys

    red, green, blue = colorsys.hsv_to_rgb(hue, 0.78, 0.95)
    return round(red * 255), round(green * 255), round(blue * 255)


def annotate_image(image: Image.Image, detections: list[dict]) -> Image.Image:
    """Draw readable class labels and confidence-aware boxes."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default(size=max(13, min(23, image.width // 48)))
    line_width = max(2, min(6, image.width // 320))
    for detection in detections:
        color = class_color(detection["class"])
        box = (
            detection["xmin"], detection["ymin"],
            detection["xmax"], detection["ymax"],
        )
        draw.rectangle(box, outline=color, width=line_width)
        label = (
            f'{detection["class"]} '
            f'{detection["confidence"]:.0%}'
        )
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        label_width = right - left
        label_height = bottom - top
        label_top = max(0, box[1] - label_height - 8)
        draw.rounded_rectangle(
            (box[0], label_top, box[0] + label_width + 12,
             label_top + label_height + 8),
            radius=5, fill=color,
        )
        draw.text(
            (box[0] + 6, label_top + 3), label,
            font=font, fill=(12, 18, 24),
        )
    return annotated


def get_groq_analysis(detections: list[dict], image_size: tuple[int, int]) -> str:
    """Ask Groq for a grounded scene brief using detection labels only."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Add it to the project-root .env file "
            "to enable the optional scene analysis."
        )
    inventory: dict[str, int] = {}
    for item in detections:
        inventory[item["class"]] = inventory.get(item["class"], 0) + 1
    completion = Groq(api_key=api_key).chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an AI vision assistant writing a concise scene "
                    "brief from object-detector output. You did not see the "
                    "image: treat the provided labels and confidence scores "
                    "as the only evidence. Do not invent people, actions, "
                    "relationships, locations, hazards, or image details. "
                    "Mention uncertainty where detections are low-confidence. "
                    "Organize the answer under Scene summary, Inventory, and "
                    "Utility and safety considerations. Safety guidance must "
                    "be general and explicitly conditional; do not infer "
                    "danger from ordinary objects alone."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Image dimensions: {image_size[0]} × {image_size[1]} px.\n"
                    f"Detection inventory and confidence: {inventory}\n"
                    f"All detections: {detections}"
                ),
            },
        ],
        temperature=0.2,
        max_tokens=450,
    )
    content = completion.choices[0].message.content
    if not content or not content.strip():
        raise RuntimeError("Groq returned an empty scene analysis.")
    return content.strip()


with st.sidebar:
    st.markdown("## Detection controls")
    confidence = st.slider(
        "Confidence threshold", min_value=0.10, max_value=0.95,
        value=0.35, step=0.05,
        help="Discard predictions below this model confidence.",
    )
    nms_threshold = st.slider(
        "NMS IoU threshold", min_value=0.10, max_value=0.90,
        value=0.45, step=0.05,
        help="Higher values allow more overlapping boxes to remain.",
    )
    st.markdown("---")
    st.markdown("### Detector")
    st.caption("YOLOv8 Nano · COCO 80 classes · CPU inference")
    st.caption("The ONNX weights are downloaded once and cached locally.")
    st.markdown("### Scene analysis")
    st.caption(f"Groq model: `{GROQ_MODEL}`")
    st.caption("Only detected labels and confidence scores are sent to Groq.")

mode = st.radio(
    "Choose image source",
    ["Sample gallery", "Upload image", "Camera snapshot"],
    horizontal=True,
    label_visibility="collapsed",
)

image: Image.Image | None = None
sample_name: str | None = None
if mode == "Sample gallery":
    sample_name = st.selectbox("Sample image", list(SAMPLES))
    st.caption(
        "Public demonstration photos are fetched on selection and cached "
        "for this session. Use the upload option for your own images."
    )
    try:
        image = open_image(fetch_sample_image(sample_name))
        st.caption(f"Sample: {sample_name} · public Ultralytics/COCO demo image")
    except (OSError, urllib.error.URLError, ValueError) as error:
        st.error(f"Could not load the selected sample image: {error}")
else:
    upload = (
        st.file_uploader(
            "Upload a JPG or PNG image",
            type=["jpg", "jpeg", "png"],
            help="Images up to 24 megapixels are supported.",
        )
        if mode == "Upload image"
        else st.camera_input("Take a camera snapshot")
    )
    if upload is not None:
        try:
            image = open_image(upload.getvalue())
        except (OSError, ValueError) as error:
            st.error(f"Could not read this image: {error}")

if image is None:
    st.info("Choose an available sample, upload an image, or take a snapshot.")
    st.stop()

left, right = st.columns([1, 1], gap="large")
with left:
    st.markdown('<div class="od-section">Input image</div>', unsafe_allow_html=True)
    st.image(image, use_container_width=True)
    st.caption(f"{image.width:,} × {image.height:,} pixels")
with right:
    st.markdown(
        '<div class="od-section">Detection preview</div>',
        unsafe_allow_html=True,
    )
    analyze = st.button(
        "Run object detection",
        type="primary",
        use_container_width=True,
        key="run-object-detection",
    )
    if analyze:
        try:
            with st.spinner("Running YOLOv8 inference…"):
                detections, inference_ms = detect_objects(
                    image, confidence, nms_threshold
                )
            st.session_state["object_detection_result"] = {
                "detections": detections,
                "inference_ms": inference_ms,
                "image_size": image.size,
                "image_bytes_hash": hashlib.sha256(
                    image.tobytes()
                ).hexdigest(),
            }
        except (OSError, urllib.error.URLError, cv2.error, RuntimeError, ValueError) as error:
            st.error(f"Object detection could not run: {error}")
            st.stop()

    result = st.session_state.get("object_detection_result")
    current_hash = hashlib.sha256(image.tobytes()).hexdigest()
    if result and result["image_bytes_hash"] == current_hash:
        detections = result["detections"]
        inference_ms = result["inference_ms"]
        annotated = annotate_image(image, detections)
        st.image(
            annotated,
            caption=f"{len(detections)} object(s) · YOLOv8 Nano",
            use_container_width=True,
        )
        average_confidence = (
            float(np.mean([item["confidence"] for item in detections]))
            if detections else 0.0
        )
        metric_columns = st.columns(3)
        metrics = (
            ("Objects", str(len(detections))),
            ("Mean confidence", f"{average_confidence:.1%}"),
            ("Inference", f"{inference_ms:.0f} ms"),
        )
        for column, (label, value) in zip(metric_columns, metrics):
            with column:
                st.markdown(
                    f'<div class="od-metric"><div class="od-metric-label">'
                    f'{label}</div><div class="od-metric-value">{value}'
                    f"</div></div>",
                    unsafe_allow_html=True,
                )
        annotated_bytes = None
        if detections:
            output = io.BytesIO()
            annotated.save(output, format="PNG", optimize=True)
            annotated_bytes = output.getvalue()
        else:
            st.info(
                "No objects met the current confidence threshold. Try "
                "lowering it or selecting another image."
            )
        if annotated_bytes:
            st.download_button(
                "Download annotated image",
                data=annotated_bytes,
                file_name="object-detection-annotated.png",
                mime="image/png",
                use_container_width=True,
            )
    else:
        st.markdown(
            '<div class="od-section"><strong>Ready to analyze</strong><br>'
            '<span class="od-muted">Run the detector to view predictions, '
            'confidence scores, and an annotated image here.</span></div>',
            unsafe_allow_html=True,
        )

if result and result["image_bytes_hash"] == hashlib.sha256(image.tobytes()).hexdigest():
    detections = result["detections"]
    st.markdown("---")
    st.subheader("Detection inventory")
    if detections:
        rows = [
            {
                "Class": item["class"],
                "Confidence": f'{item["confidence"]:.1%}',
                "ymin": item["ymin"],
                "xmin": item["xmin"],
                "ymax": item["ymax"],
                "xmax": item["xmax"],
            }
            for item in detections
        ]
        with st.expander(f"View {len(rows)} detected object(s)", expanded=True):
            st.dataframe(rows, hide_index=True, use_container_width=True)
    else:
        st.caption("The detector found no objects above the selected threshold.")

    st.subheader("AI scene analysis")
    st.caption(
        "The language model summarizes detector output only; it does not "
        "receive or visually inspect the uploaded image."
    )
    if st.button("Generate Groq scene brief", key="generate-scene-brief"):
        try:
            with st.spinner(f"Generating a grounded brief with {GROQ_MODEL}…"):
                analysis = get_groq_analysis(detections, image.size)
            st.session_state["object_detection_analysis"] = {
                "image_bytes_hash": result["image_bytes_hash"],
                "text": analysis,
            }
        except (GroqError, RuntimeError, ValueError) as error:
            st.error(f"Groq scene analysis failed: {error}")
    analysis_result = st.session_state.get("object_detection_analysis")
    if analysis_result and analysis_result["image_bytes_hash"] == result["image_bytes_hash"]:
        st.markdown(analysis_result["text"])

st.markdown("---")
st.caption(
    "Detector weights: YOLOv8 Nano ONNX from Ultralytics. Sample images are "
    "public Ultralytics and COCO demonstration images. Detection quality "
    "depends on image quality, object size, and confidence settings."
)
