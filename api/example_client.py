 """
Example client to test the streaming API endpoint.
This demonstrates how to consume the Server-Sent Events (SSE) from the /api/explain/stream endpoint.
"""

import requests
import json
from pathlib import Path


def _base_api_url() -> str:
    config_path = Path(__file__).resolve().parents[2] / "config.json"
    root_config = json.loads(config_path.read_text())
    backend_cfg = root_config["backend"]
    host = backend_cfg["host"]
    port = backend_cfg["port"]
    return f"http://{host}:{port}/api"


def test_streaming_endpoint():
    """Test the streaming explain endpoint"""
    url = f"{_base_api_url()}/explain/stream"
    
    payload = {
        "user_input": "What is photosynthesis?",
        "messages": [],
        "learner": {
            "age": 10
        },
        "intent": "new_question"
    }
    
    print("🚀 Sending request to streaming endpoint...")
    print(f"📝 Question: {payload['user_input']}\n")
    
    with requests.post(url, json=payload, stream=True) as response:
        print("📡 Receiving streaming response:\n")
        print("-" * 60)
        
        for line in response.iter_lines():
            if line:
                # Decode the line
                decoded_line = line.decode('utf-8')
                
                # SSE format: "data: {json}"
                if decoded_line.startswith('data: '):
                    data_str = decoded_line[6:]  # Remove "data: " prefix
                    data = json.loads(data_str)
                    
                    if data.get('event') == 'complete':
                        print("\n✅ Stream completed!")
                        break
                    elif data.get('event') == 'error':
                        print(f"\n❌ Error: {data.get('error')}")
                        break
                    else:
                        # Print the node and its output
                        node = data.get('node')
                        print(f"\n🔄 Node: {node}")
                        
                        # Print relevant fields from the node output
                        node_data = data.get('data', {})
                        if node_data.get('final_output'):
                            print(f"📤 Final Output: {json.dumps(node_data['final_output'], indent=2)}")
        
        print("-" * 60)


def test_non_streaming_endpoint():
    """Test the non-streaming explain endpoint"""
    url = f"{_base_api_url()}/explain"
    
    payload = {
        "user_input": "What is photosynthesis?",
        "messages": [],
        "learner": {
            "age": 10
        },
        "intent": "new_question"
    }
    
    print("\n🚀 Sending request to non-streaming endpoint...")
    print(f"📝 Question: {payload['user_input']}\n")
    
    response = requests.post(url, json=payload)
    result = response.json()
    
    print("📡 Complete response:\n")
    print("-" * 60)
    print(json.dumps(result, indent=2))
    print("-" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("🧪 AgeXplain API Test Client")
    print("=" * 60)
    
    # Test streaming endpoint
    try:
        test_streaming_endpoint()
    except Exception as e:
        print(f"❌ Streaming test failed: {e}")
    
    # Test non-streaming endpoint
    # Uncomment to test:
    # try:
    #     test_non_streaming_endpoint()
    # except Exception as e:
    #     print(f"❌ Non-streaming test failed: {e}")
