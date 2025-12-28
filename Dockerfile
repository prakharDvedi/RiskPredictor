# Use Python 3.12
FROM python:3.12-slim-bookworm

# Set up a new user named "user" with user ID 1000
# Hugging Face Spaces requires this specific user setup for security
RUN useradd -m -u 1000 user

# Set working directory and permissions
WORKDIR /app

# Install system tools
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Switch to the "user" context
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Copy requirements first (for caching)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- CRITICAL: BAKE DATA INTO THE IMAGE ---
# We copy the 'src', 'data', and 'outputs' folders inside.
# This makes the container fully autonomous.
COPY --chown=user src ./src
COPY --chown=user data ./data
COPY --chown=user outputs ./outputs

# Expose the port Hugging Face expects (7860)
EXPOSE 7860

# Launch Command
# Note: server.port must be 7860 and address must be 0.0.0.0
CMD ["streamlit", "run", "src/frontend/app.py", "--server.port=7860", "--server.address=0.0.0.0"]