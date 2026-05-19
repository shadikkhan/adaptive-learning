#!/usr/bin/env python3
"""
Start both FastAPI backend and Streamlit frontend from a single command.
Reads ports from config.json.
"""

import json
import subprocess
import sys
import time
import socket
from pathlib import Path


def wait_for_port(host, port, service_name, timeout=30, retries=60):
    """
    Wait for a TCP port to be ready by attempting to connect.
    Returns True if ready, False if timeout.
    """
    start_time = time.time()
    attempt = 0
    
    while attempt < retries:
        try:
            sock = socket.create_connection((host, port), timeout=1)
            sock.close()
            elapsed = time.time() - start_time
            print(f"      ✓ {service_name} is ready! (took {elapsed:.1f}s)")
            return True
        except (socket.timeout, socket.error):
            attempt += 1
            elapsed = time.time() - start_time
            
            if elapsed > timeout:
                print(f"      ✗ {service_name} failed to start (timeout after {timeout}s)")
                return False
            
            # Show progress every 5 attempts
            if attempt % 5 == 0:
                print(f"      ⏳ Waiting for {service_name}... ({elapsed:.1f}s)")
            
            time.sleep(0.5)
    
    return False


def main():
    config_path = Path(__file__).parent / "config.json"
    if not config_path.exists():
        print(f"Error: config.json not found at {config_path}")
        sys.exit(1)
    
    with open(config_path) as f:
        config = json.load(f)
    
    backend_host = config.get("backend", {}).get("host", "127.0.0.1")
    backend_port = config.get("backend", {}).get("port", 8001)
    frontend_port = config.get("frontend", {}).get("dev_port", 8501)
    
    print("=" * 70)
    print("AgeXplain Adaptive Learning - Starting Backend & Frontend")
    print("=" * 70)
    print(f"\n📍 Backend:  http://{backend_host}:{backend_port}")
    print(f"📍 Frontend: http://localhost:{frontend_port}")
    print(f"📍 API Docs: http://{backend_host}:{backend_port}/docs")
    print("\n" + "=" * 70)
    print("Press Ctrl+C to stop both services")
    print("=" * 70 + "\n")
    
    processes = []
    
    try:
        # Start backend in background
        print(f"[1/2] Starting FastAPI backend on {backend_host}:{backend_port}...")
        backend_proc = subprocess.Popen(
            [sys.executable, "main.py"],
            cwd=Path(__file__).parent,
            stdout=None,
            stderr=None,
        )
        processes.append(("Backend", backend_proc))
        print("      ✓ Backend process started")
        
        # Wait for backend to be ready
        print(f"      ⏳ Waiting for backend to listen on {backend_host}:{backend_port}...")
        if not wait_for_port(backend_host, backend_port, "Backend"):
            print("      ✗ Backend failed to start!")
            backend_proc.terminate()
            sys.exit(1)
        
        # Start frontend
        print(f"\n[2/2] Starting Streamlit frontend on port {frontend_port}...")
        frontend_proc = subprocess.Popen(
            [sys.executable, "run_streamlit.py"],
            cwd=Path(__file__).parent,
            stdout=None,
            stderr=None,
        )
        processes.append(("Frontend", frontend_proc))
        print("      ✓ Frontend process started")

        # Wait for frontend to be ready
        print(f"      ⏳ Waiting for frontend to listen on 127.0.0.1:{frontend_port}...")
        if not wait_for_port("127.0.0.1", frontend_port, "Frontend"):
            print("      ✗ Frontend failed to start!")
            frontend_proc.terminate()
            backend_proc.terminate()
            sys.exit(1)

        print()
        
        print("=" * 70)
        print("✅ Both services are running!")
        print("=" * 70 + "\n")
        
        # Monitor both processes
        while True:
            for name, proc in processes:
                if proc.poll() is not None:  # Process has exited
                    print(f"\n⚠️  {name} process exited with code {proc.returncode}")
                    raise KeyboardInterrupt
            
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("Shutting down services...")
        print("=" * 70)
        
        for name, proc in processes:
            if proc.poll() is None:  # Process is still running
                print(f"Stopping {name}...")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                    print(f"  ✓ {name} stopped")
                except subprocess.TimeoutExpired:
                    print(f"  Force killing {name}...")
                    proc.kill()
        
        print("\n✓ All services stopped\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
