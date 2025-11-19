import os
import json
import requests
import gradio as gr

# Base URL for Ollama (change if you run it on a different host/port)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")


def list_models():
    """
    Fetch available models from Ollama.
    Returns a list of model names.
    """
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        models = [m["name"] for m in data.get("models", [])]
        if not models:
            # Fallback placeholder if no models found
            return ["llama3"]
        return models
    except Exception as e:
        print(f"Error fetching model list from Ollama: {e}")
        # Fallback to a default; adjust to whatever you actually have pulled
        return ["llama3"]


def stream_chat_ollama(model: str, messages):
    """
    Generator that streams chat responses from Ollama.
    messages = list of {role: "user"/"assistant"/"system", content: "..."} dicts
    """
    url = f"{OLLAMA_URL}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
    }

    with requests.post(url, json=payload, stream=True) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue

            # Each chunk has "message": {"role": "assistant", "content": "..."}, plus possibly "done"
            msg = data.get("message", {})
            content = msg.get("content", "")
            if content:
                yield content

            if data.get("done"):
                break


def build_messages(history, user_message, system_prompt):
    """
    Convert Gradio chat history + current user message into Ollama message format.
    history: list of [user, assistant] pairs.
    """
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Existing turns
    for user, assistant in history:
        messages.append({"role": "user", "content": user})
        if assistant:
            messages.append({"role": "assistant", "content": assistant})

    # Current user message
    messages.append({"role": "user", "content": user_message})

    return messages


def chat_fn(user_message, history, model_name, system_prompt):
    """
    Gradio callback for the chat.
    This is a generator to support streaming output.
    """
    if history is None:
        history = []

    # Build messages for Ollama from EXISTING history (without current turn)
    messages = build_messages(history, user_message, system_prompt)

    # Add the user's new message with empty assistant response
    history = history + [[user_message, ""]]

    # Stream chunks from Ollama and update the last assistant message
    partial_response = ""
    try:
        for chunk in stream_chat_ollama(model_name, messages):
            partial_response += chunk
            history[-1][1] = partial_response
            # Yield updated history; Gradio will update the UI incrementally
            yield history
    except Exception as e:
        error_text = f"[Error talking to Ollama: {e}]"
        history[-1][1] = error_text
        yield history


def main():
    models = list_models()
    default_model = models[0] if models else "llama3"

    with gr.Blocks(title="Local ChatGPT (Ollama)") as demo:
        gr.Markdown(
            """
            # Local ChatGPT Clone (Ollama + Gradio)

            - **Backend:** Ollama running locally  
            - **UI:** Gradio  
            - **Features:** Model selection, chat history, streaming output
            """
        )

        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=models,
                value=default_model,
                label="Model",
                interactive=True,
            )
            system_prompt = gr.Textbox(
                label="System Prompt (optional)",
                lines=2,
                placeholder="e.g. You are a concise, helpful assistant.",
                value="You are a helpful assistant running locally.",
            )

        chatbot = gr.Chatbot(label="Chat")
        msg = gr.Textbox(
            label="Your message",
            placeholder="Type a message and press Enter...",
        )
        clear_btn = gr.Button("Clear")

        # When the user hits Enter in the message box:
        msg.submit(
            chat_fn,
            inputs=[msg, chatbot, model_dropdown, system_prompt],
            outputs=chatbot,
        )

        # Also clear the textbox after sending
        msg.submit(lambda: "", None, msg)

        # Clear button resets the chat
        clear_btn.click(
            lambda: [],
            outputs=chatbot,
        )
    # Important: disable Gradio's API schema generation to avoid json_schema bug
        # enable queue so generator (streaming) functions work
    demo.queue()

    # launch the app
    demo.launch()
    

if __name__ == "__main__":
    main()
