#!/usr/bin/env python3
"""Launch Streamlit frontend with port from config.json."""

import json
import subprocess
import sys
from pathlib import Path


def main():
    config_path = Path(__file__).parent / "config.json"
    if not config_path.exists():
        print(f"Error: config.json not found at {config_path}")
        sys.exit(1)
    
    with open(config_path) as f:
        config = json.load(f)
    
    port = config.get("frontend", {}).get("dev_port", 8501)
    print(f"Starting Streamlit on port {port} (configured in config.json)...")
    
    cmd = ["streamlit", "run", "app.py", "--server.port", str(port)]
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
