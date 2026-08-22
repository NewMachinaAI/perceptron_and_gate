"""REST inference service for a Perceptron saved by perceptron_and_train.py.

Loads model.safetensors + config.json from a model directory and serves a
GET endpoint that predicts the AND-gate output for two input values.

Example:
    GET /predict?a=1&b=0
    -> {"a": 1.0, "b": 0.0, "result": 0, "probability": 0.0889}
"""

import argparse
import json
import os

import torch
from flask import Flask, jsonify, request
from safetensors.torch import load_file

from container_train_src.perceptron_and_train import Perceptron, OUT_DIR

app = Flask(__name__)
model: Perceptron = None


def load_model(model_dir: str) -> Perceptron:
    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)

    loaded = Perceptron(in_features=config["in_features"])
    state_dict = load_file(os.path.join(model_dir, "model.safetensors"))
    loaded.load_state_dict(state_dict)
    loaded.eval()
    return loaded


def predict(a: float, b: float):
    with torch.no_grad():
        x = torch.tensor([[a, b]], dtype=torch.float32)
        prob = model(x).item()
        label = 1 if prob >= 0.5 else 0
    return prob, label


@app.get("/predict")
def predict_endpoint():
    a_raw = request.args.get("a")
    b_raw = request.args.get("b")
    if a_raw is None or b_raw is None:
        return jsonify(error="query parameters 'a' and 'b' are required"), 400

    try:
        a = float(a_raw)
        b = float(b_raw)
    except ValueError:
        return jsonify(error="'a' and 'b' must be numbers"), 400

    prob, label = predict(a, b)
    return jsonify(a=a, b=b, result=label, probability=round(prob, 4))


def main():
    parser = argparse.ArgumentParser(description="Serve REST inference for a trained AND-gate Perceptron.")
    parser.add_argument(
        "--model-dir",
        default=OUT_DIR,
        help=f"Directory containing model.safetensors and config.json (default: {OUT_DIR})",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5001, help="Port to bind (default: 5001)")
    args = parser.parse_args()

    global model
    model = load_model(args.model_dir)

    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
