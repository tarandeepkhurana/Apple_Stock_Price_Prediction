#Use python base image
FROM python:3.10-slim

#Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

#Set working directory
WORKDIR /app

# Copy everything
COPY . /app/

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port your app runs on
EXPOSE 5002

# Run the Flask app
CMD ["python", "app.py"]