"""Main entry point placeholder for the IPL Agentic RAG Gradio application."""

import argparse


def main() -> None:
    """Parse input arguments and print a placeholder startup message."""
    parser = argparse.ArgumentParser(description="IPL Agentic RAG starter app")
    parser.add_argument("--question", type=str, default="", help="Question string for testing")
    args = parser.parse_args()

    print("IPL Agentic RAG skeleton is ready. Gradio app logic is not implemented yet.")
    if args.question:
        print(f"Received question placeholder: {args.question}")


if __name__ == "__main__":
    main()
