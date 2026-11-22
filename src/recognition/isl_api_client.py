
import json
import urllib.error
import urllib.request

API_BASE_URL = "https://creator-090-isl-api.hf.space"


def check_api_health():
    """Check whether the hosted ISL model is ready."""
    url = f"{API_BASE_URL}/health"

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.load(response)

    except urllib.error.HTTPError as error:
        raise RuntimeError(
            f"API returned HTTP {error.code}"
        ) from error

    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Could not connect to ISL API: {error.reason}"
        ) from error


if __name__ == "__main__":
    print("Checking GestureBridge ISL API connection...")

    try:
        result = check_api_health()
        print(json.dumps(result, indent=2))

        if result.get("model_loaded"):
            print("SUCCESS: ISL model is ready!")
        else:
            print("The API responded, but the model is not ready.")

    except Exception as error:
        print(f"Connection failed: {error}")
