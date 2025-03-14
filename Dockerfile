# Use an official Python image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy files
COPY . /app

# Create cache directory and set permissions
RUN mkdir -p /app/cache && chmod -R 777 /app/cache

# Set environment variables for cache
ENV TRANSFORMERS_CACHE=/app/cache
ENV HF_HOME=/app/cache

# Install dependencies
RUN pip install -r requirements.txt

# Expose port (default Hugging Face Spaces port)
EXPOSE 7860

# Run the Flask app with Gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:7860", "app:app"]
