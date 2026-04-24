from dotenv import load_dotenv

load_dotenv()

from app.gradio_app import build_app

if __name__ == "__main__":
    import os

    app = build_app()
    app.launch(
        server_port=int(os.getenv("GRADIO_PORT", 7860)),
        share=os.getenv("GRADIO_SHARE", "false").lower() == "true",
        show_error=True,
    )
