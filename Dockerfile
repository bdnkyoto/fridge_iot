# syntax=docker/dockerfile:1

# ---- Builder Stage ----
FROM python:3.13 as builder

    # Install build-time system dependencies + runtime ones needed for installation
RUN apt-get update && apt-get install -y --no-install-recommends \
        # Build tools if needed (e.g., gcc, build-essential)
        # Any -dev packages needed to BUILD python libs (like libxrender-dev?)
    libxrender-dev \
        # Runtime libs also needed here if pip install checks for them
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

    # Install Python dependencies using a cache mount for rebuilds
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

    # ---- Final Stage ----
FROM python:3.13-slim

    # Install ONLY runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
        # Notice -dev package is NOT installed here
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

    # Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

    # Copy application code (after dependencies)
COPY . .

    # Create necessary directories
RUN mkdir -p instance/user_uploads/snapshots

    # Expose the port the app runs on
EXPOSE 5000

    # Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

    # Volume for persistent data
VOLUME ["/app/instance"]

    # Command to run the application
CMD ["python", "app.py"]
