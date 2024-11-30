# Use Python 3.10 as the base image to support Gradio
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the application and dependencies into the container
COPY ./app /app/app
COPY ./infra/requirements.txt /app/requirements.txt

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port for Gradio
EXPOSE 7860

# Command to run the Gradio app
CMD ["python", "app/app.py", "--server-name", "0.0.0.0", "--server-port", "7860"]
