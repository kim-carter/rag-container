FROM ollama/ollama:latest

# Copy an entrypoint script to handle the server and model pull
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose the default Ollama API port
EXPOSE 11434

# Use the custom entrypoint script
ENTRYPOINT ["/entrypoint.sh"]


