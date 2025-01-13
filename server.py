#!/usr/bin/env python3

from llama_cpp import Llama
import uvicorn
from llama_cpp.server.app import create_app
from llama_cpp.server.settings import Settings


def main():
    print("Starting server...")

    # Configure model settings
    settings = Settings(
        model="./arcee-agent.Q4_K_M.gguf",
        n_ctx=2048,                # Context window size
        n_gpu_layers=-1,           # Use all GPU layers
        n_threads=4,               # Number of CPU threads
        embedding=True,            # Enable embeddings
        verbose=True               # Show detailed logs
    )

    # Create FastAPI app
    app = create_app(settings)

    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )


if __name__ == "__main__":
    main()
