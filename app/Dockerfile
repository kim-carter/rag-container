# Base image
FROM python:3.9-slim

# Install required system packages
RUN apt-get update && apt-get install -y libglib2.0-0 vim telnet curl && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only the requirements file first (for caching)
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the application files
COPY . /app

# Install the spaCy model (this is needed after dependencies are installed)
RUN python -m spacy download en_core_web_sm

# Expose the Ollama port
EXPOSE 11434

# Command to run the application
CMD ["python", "main.py"]

