FROM python:3.10-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files and application code
COPY .pkl/ .pkl/
COPY streamlit/ streamlit/
COPY src/ src/
COPY data/ data/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose the port Streamlit runs on
EXPOSE 8501

# Set the entry point
CMD ["streamlit", "run", "streamlit/Home.py", "--server.address=0.0.0.0"] 