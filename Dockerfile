# Stage 1: Build Frontend React App
FROM node:18-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy package files first for better caching
COPY frontend-react/package*.json ./

# Install dependencies (including dev dependencies for build)
RUN npm ci --silent

# Copy frontend source code
COPY frontend-react/ ./

# Build React app with proper permissions
RUN chmod +x node_modules/.bin/* && \
    npm run build

# Stage 2: Python Backend + Serve Frontend
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY api/ ./api/
COPY core/ ./core/

# Copy built frontend from stage 1
COPY --from=frontend-builder /app/frontend/build ./frontend-react/build

# Create directories for uploads and logs
RUN mkdir -p uploads logs

# Expose port (Azure Container Apps default)
EXPOSE 8080

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8080

# Health check using curl (no external dependencies)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application on port 8080 (Azure Container Apps standard)
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port $PORT"]