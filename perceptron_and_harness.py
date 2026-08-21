"""CLI harness that prompts for A and B, calls the perceptron_and_inference.py
REST endpoint, and prints the result in a friendly format.
"""

import argparse
import json
import urllib.error
import urllib.parse
import urllib.request


def prompt_float(prompt: str) -> float:
    while True:
        raw = input(prompt).strip()
        try:
            return float(raw)
        except ValueError:
            print("Please enter a number (e.g. 0 or 1).")


def call_predict(base_url: str, a: float, b: float) -> dict:
    query = urllib.parse.urlencode({"a": a, "b": b})
    url = f"{base_url}/predict?{query}"
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read())


def main():
    parser = argparse.ArgumentParser(description="CLI harness for the AND-gate Perceptron REST service.")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:5000",
        help="Base URL of the perceptron_and_inference.py service (default: http://127.0.0.1:5000)",
    )
    args = parser.parse_args()
    base_url = args.url.rstrip("/")

    a = prompt_float("Enter value A: ")
    b = prompt_float("Enter value B: ")

    try:
        result = call_predict(base_url, a, b)
    except urllib.error.HTTPError as e:
        body = json.loads(e.read())
        print(f"Request failed ({e.code}): {body.get('error', e.reason)}")
        return
    except urllib.error.URLError as e:
        print(f"Could not reach the inference service at {base_url}: {e.reason}")
        print("Is perceptron_and_inference.py running?")
        return

    label = "TRUE" if result["result"] == 1 else "FALSE"
    print(f"\nAND({result['a']}, {result['b']}) = {label}")
    print(f"Confidence: {result['probability'] * 100:.2f}%")


if __name__ == "__main__":
    main()
