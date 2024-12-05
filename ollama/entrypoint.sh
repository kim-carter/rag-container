#!/bin/bash

# Start the Ollama server in the background
ollama serve &

# Wait for the server to start
echo "Waiting for Ollama server to start..."
sleep 5

# Pull the required model
echo "Pulling model mistral-7b-instruct-q4_K_M..."
ollama pull mistral:7b-instruct-q4_K_M
ollama cp mistral:7b-instruct-q4_K_M ms7b

# Wait for the server to remain active
echo "Ollama server is running and model is ready."
wait

