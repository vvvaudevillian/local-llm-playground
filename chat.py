import requests

OLLAMA_URL = "http://127.0.0.1:11434"
MODEL = "llama3.1:8b"

def generate(prompt: str) -> str:
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "").strip()

if __name__ == "__main__":
    print(generate("Explain verifiable credentials in one short paragraph."))
