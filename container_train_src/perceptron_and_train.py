"""Trains a single-layer Perceptron on the AND logic gate, saves it (safetensors +
config.json), and tests it against the full truth table."""

import json
import os

import torch
import torch.nn as nn
from safetensors.torch import save_file

MODEL_TYPE = "perceptron"
OUT_DIR = "and_perceptron_model"


class Perceptron(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.linear(x))


def train():
    torch.manual_seed(0)

    # AND gate truth table
    X = torch.tensor(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    )
    y = torch.tensor([[0.0], [0.0], [0.0], [1.0]])

    model = Perceptron(in_features=2)
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    epochs = 2000
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 400 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    return model


def predict(model, a: float, b: float):
    model.eval()
    with torch.no_grad():
        x = torch.tensor([[a, b]], dtype=torch.float32)
        prob = model(x).item()
        label = 1 if prob >= 0.5 else 0
    return prob, label


def save_pretrained(model: Perceptron, out_dir: str, in_features: int):
    """Save weights as safetensors and architecture as config.json, Hugging Face style."""
    os.makedirs(out_dir, exist_ok=True)

    state_dict = model.state_dict()
    save_file(state_dict, os.path.join(out_dir, "model.safetensors"))

    config = {
        "model_type": MODEL_TYPE,
        "architectures": ["Perceptron"],
        "in_features": in_features,
        "out_features": 1,
        "activation": "sigmoid",
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved model.safetensors and config.json to {out_dir}/")


if __name__ == "__main__":
    model = train()

    print("\nInference results (AND gate):")
    for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        prob, label = predict(model, a, b)
        print(f"AND({a}, {b}) = {label}  (probability: {prob:.4f})")

    save_pretrained(model, OUT_DIR, in_features=2)
