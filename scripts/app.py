import os
import time
import pickle
from collections import Counter, deque

import numpy as np
from flask import Flask, send_from_directory
from flask_socketio import SocketIO, emit

try:
    from tensorflow.keras.models import load_model
except Exception:  # pragma: no cover
    load_model = None


SEQUENCE_LENGTH = int(os.environ.get("SEQUENCE_LENGTH", "20"))
DEFAULT_CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.85"))
STABILITY_WINDOW = int(os.environ.get("STABILITY_WINDOW", "10"))
STABILITY_MIN_HITS = int(os.environ.get("STABILITY_MIN_HITS", "7"))

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join("models", "sign_lstm_model.h5"))
LABEL_PATH = os.environ.get("LABEL_PATH", os.path.join("models", "labels.pkl"))


def _load_label_encoder(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_model(path: str):
    if load_model is None:
        raise RuntimeError("TensorFlow/Keras not available in this environment.")
    return load_model(path)


def _softmax_argmax(probs: np.ndarray):
    class_id = int(np.argmax(probs))
    conf = float(probs[class_id])
    return class_id, conf


app = Flask(__name__, static_folder=None)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


model = None
label_encoder = None
model_load_error = None

try:
    model = _load_model(MODEL_PATH)
    label_encoder = _load_label_encoder(LABEL_PATH)
except Exception as e:  # keep server running for UI demos
    model_load_error = str(e)


client_state = {}


def _get_state(sid: str):
    st = client_state.get(sid)
    if st is None:
        st = {
            "sequence": deque(maxlen=SEQUENCE_LENGTH),
            "stability": deque(maxlen=STABILITY_WINDOW),
            "last_word": None,
            "last_emit_ts": 0.0,
            "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
        }
        client_state[sid] = st
    return st


WEB_DIR = os.path.join(os.path.dirname(__file__), "..", "Interface")


@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.get("/<path:filename>")
def web_static(filename: str):
    # Serve other static assets (e.g. main.js) with correct MIME type so they load on localhost.
    mimetype = None
    if filename.endswith(".js"):
        mimetype = "application/javascript"
    return send_from_directory(WEB_DIR, filename, mimetype=mimetype)


@socketio.on("connect")
def on_connect():
    st = _get_state(request_sid())
    emit(
        "server_status",
        {
            "ok": model is not None and label_encoder is not None,
            "model_path": MODEL_PATH,
            "label_path": LABEL_PATH,
            "error": model_load_error,
            "sequence_length": SEQUENCE_LENGTH,
            "stability_window": STABILITY_WINDOW,
            "stability_min_hits": STABILITY_MIN_HITS,
            "confidence_threshold": st["confidence_threshold"],
        },
    )


@socketio.on("disconnect")
def on_disconnect():
    client_state.pop(request_sid(), None)


@socketio.on("set_settings")
def on_set_settings(payload):
    st = _get_state(request_sid())
    if isinstance(payload, dict):
        if "confidence_threshold" in payload:
            try:
                st["confidence_threshold"] = float(payload["confidence_threshold"])
            except Exception:
                pass
    emit("settings", {"confidence_threshold": st["confidence_threshold"]})


@socketio.on("landmarks")
def on_landmarks(payload):
    """
    payload:
      {
        "ts": <ms epoch>,
        "landmarks_63": [63 floats]  # normalized, wrist-centered, scaled
      }
    """
    st = _get_state(request_sid())
    landmarks = None
    if isinstance(payload, dict):
        landmarks = payload.get("landmarks_63")
    if not isinstance(landmarks, list) or len(landmarks) != 63:
        emit("prediction", {"ok": False, "error": "Expected landmarks_63 with 63 floats."})
        return

    try:
        st["sequence"].append([float(x) for x in landmarks])
    except Exception:
        emit("prediction", {"ok": False, "error": "Invalid landmark values."})
        return

    if len(st["sequence"]) < SEQUENCE_LENGTH:
        emit(
            "prediction",
            {
                "ok": True,
                "ready": False,
                "word": "...",
                "confidence": 0.0,
            },
        )
        return

    if model is None or label_encoder is None:
        emit(
            "prediction",
            {
                "ok": False,
                "ready": True,
                "word": "...",
                "confidence": 0.0,
                "error": f"Model not loaded: {model_load_error}",
            },
        )
        return

    input_data = np.expand_dims(list(st["sequence"]), axis=0)
    probs = model.predict(input_data, verbose=0)[0]
    class_id, conf = _softmax_argmax(probs)

    st["stability"].append(class_id)
    stable_id = None
    if len(st["stability"]) == STABILITY_WINDOW:
        stable_id = Counter(st["stability"]).most_common(1)[0][0]

    word = "..."
    is_stable = (
        stable_id is not None
        and st["stability"].count(stable_id) >= STABILITY_MIN_HITS
        and conf >= st["confidence_threshold"]
    )

    if is_stable:
        try:
            word = label_encoder.inverse_transform([stable_id])[0]
        except Exception:
            word = str(stable_id)

    emit(
        "prediction",
        {
            "ok": True,
            "ready": True,
            "word": word,
            "class_id": class_id,
            "confidence": conf,
            "is_stable": bool(is_stable),
            "confidence_threshold": st["confidence_threshold"],
        },
    )

    if is_stable and word != st["last_word"]:
        st["last_word"] = word
        st["last_emit_ts"] = time.time()
        st["sequence"].clear()


def request_sid() -> str:
    # Import here to keep module import side-effects minimal.
    from flask import request

    return request.sid


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    socketio.run(app, host="0.0.0.0", port=port, debug=True)
