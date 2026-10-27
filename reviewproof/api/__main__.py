"""Run the API (and the built frontend, if frontend/dist exists): python -m reviewproof.api"""
import argparse

import uvicorn
from dotenv import load_dotenv

from reviewproof.api.app import create_app

if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run the ReviewProof API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port)
