import requests
import json

def generate_response_with_stream(query, context, ollama_url, model_name="ms7b", timeout=30):
    """
    Generate a response using Ollama via a streaming connection.

    Args:
        query (str): The user query.
        context (str): Context for the query.
        ollama_url (str): The base URL for the Ollama backend.
        model_name (str): The name of the model to use.
        timeout (int): Timeout for the HTTP request in seconds.

    Returns:
        str: The generated response.
    """
    # Prepare the payload
    data = {
        "model": model_name,
        "prompt": f"Use the following context to answer the question:\n\nContext:\n{context}\n\nQuestion: {query}",
    }

    try:
        # Send POST request to Ollama with streaming enabled
        with requests.post(f"{ollama_url}/api/generate", json=data, stream=True, timeout=timeout) as response:
            if response.status_code != 200:
                return f"Error: Received status code {response.status_code}.\nResponse: {response.text}"

            print("Streaming response from Ollama...")
            accumulated_response = ""

            # Process each line in the streamed response
            for line in response.iter_lines():
                if line:
                    try:
                        # Parse the JSON line
                        json_object = json.loads(line.decode("utf-8"))
                        print(f"Received JSON: {json_object}")  # Debugging

                        # Accumulate response content
                        accumulated_response += json_object.get("response", "")

                        # Stop if done:true is encountered
                        if json_object.get("done", False):
                            break
                    except json.JSONDecodeError as e:
                        print(f"Error decoding line: {line}. Error: {e}")
                        continue

            return accumulated_response.strip() if accumulated_response else "No meaningful response generated."

    except requests.ConnectionError as e:
        return f"ConnectionError: Unable to reach Ollama at {ollama_url}. Details: {e}"
    except requests.Timeout as e:
        return f"TimeoutError: Ollama did not respond in time. Details: {e}"

