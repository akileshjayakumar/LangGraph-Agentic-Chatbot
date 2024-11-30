# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the requirements.txt into the container
COPY infra/requirements.txt /app/requirements.txt

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Copy the application code into the container
COPY app /app/app

# Set the command to run the app
CMD ["python", "/app/app/app.py"]
