# Use official Python image
FROM python:3.9.13

# Set working directory
WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Command to run your script
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]