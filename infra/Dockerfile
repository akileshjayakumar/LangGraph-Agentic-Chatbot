# Base image with Python
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy application and infrastructure files
COPY ./app /app/app
COPY ./infra/requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port for Gradio
EXPOSE 7860

# Command to run the Gradio app
CMD ["python", "app/app.py", "--server-name", "0.0.0.0", "--server-port", "7860"]
