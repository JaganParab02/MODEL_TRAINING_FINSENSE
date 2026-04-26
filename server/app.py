import uvicorn
import gradio as gr
from finsense.server import app as fastapi_app
from assistant_ui import build_ui

# Build the Gradio UI
demo = build_ui()

# Mount Gradio onto the existing FastAPI app at the root path.
# This serves the UI when users visit the Hugging Face Space URL.
# The backend API endpoints (/reset, /step, etc.) remain fully functional.
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

def main():
    """Main entry point for the OpenEnv server and UI."""
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
