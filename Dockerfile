# Start with a base Python image
FROM python:3.9-slim

# Install dependencies for OpenCV, including libGL
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the handler script into the container
COPY app /app

# Command to run the handler script
CMD ["python", "handler.py"]
