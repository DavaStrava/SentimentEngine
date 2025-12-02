# Deployment Guide

## Overview

The SentimentEngine Streamlit application can be deployed as a containerized application for easy sharing and demonstration of the PoC interface.

## Deployment Script

The deployment process is automated via `scripts/deploy.sh`. This script:

1. Builds a Docker container with the Streamlit application
2. Pushes the container to a registry (if configured)
3. Deploys to the target environment
4. Outputs the deployment URL

## Manual Deployment Trigger

Use the "Deploy Streamlit Application" hook from the command palette or hooks panel to trigger deployment manually. This is the recommended approach rather than automatic deployment on merge, as it gives you control over when deployments occur.

## Container Configuration

### Dockerfile Structure

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    redis-server \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY models/ ./models/

# Expose Streamlit port
EXPOSE 8501

# Start Redis in background and run Streamlit
CMD redis-server --daemonize yes && streamlit run src/ui/display.py --server.port=8501 --server.address=0.0.0.0
```

## Environment Variables

Configure these environment variables for deployment:

- `SENTIMENT_ENV`: Environment name (development, testing, production)
- `REDIS_URL`: Redis connection URL (default: redis://localhost:6379)
- `STREAMLIT_PORT`: Port for Streamlit server (default: 8501)
- `MODEL_PATH`: Base path for pre-trained models (default: ./models/)
- `LOG_LEVEL`: Logging level (default: INFO)

## Deployment Script Template

Create `scripts/deploy.sh`:

```bash
#!/bin/bash
set -e

echo "Building SentimentEngine container..."
docker build -t sentimentengine:latest .

echo "Tagging container..."
docker tag sentimentengine:latest ${REGISTRY_URL}/sentimentengine:latest

echo "Pushing to registry..."
docker push ${REGISTRY_URL}/sentimentengine:latest

echo "Deploying to target environment..."
# Add your deployment commands here (kubectl, docker-compose, cloud provider CLI, etc.)

echo "Deployment complete!"
echo "Application URL: ${DEPLOYMENT_URL}"
```

## Local Testing

Test the container locally before deployment:

```bash
# Build the container
docker build -t sentimentengine:latest .

# Run locally
docker run -p 8501:8501 -v $(pwd)/models:/app/models sentimentengine:latest

# Access at http://localhost:8501
```

## Cloud Deployment Options

### Option 1: Docker Compose (Simple)

```yaml
version: '3.8'
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
  
  sentimentengine:
    image: sentimentengine:latest
    ports:
      - "8501:8501"
    environment:
      - REDIS_URL=redis://redis:6379
      - SENTIMENT_ENV=production
    depends_on:
      - redis
    volumes:
      - ./models:/app/models
```

### Option 2: Kubernetes

Create deployment manifests in `k8s/` directory with appropriate service, deployment, and ingress configurations.

### Option 3: Cloud Platform (AWS/GCP/Azure)

Use platform-specific container services:
- AWS: ECS/Fargate or App Runner
- GCP: Cloud Run or GKE
- Azure: Container Instances or AKS

## Model Files

Ensure pre-trained model files are available in the container:
- Include models in the Docker image (increases image size)
- Mount models as a volume (recommended for large models)
- Download models at container startup (requires internet access)

See `model-acquisition.md` for details on obtaining pre-trained models.

## Security Considerations

- Never commit sensitive credentials to the repository
- Use secrets management for API keys and passwords
- Configure authentication for production deployments
- Use HTTPS/TLS for production URLs
- Restrict network access to Redis

## Monitoring

Add health check endpoints:

```python
# In src/ui/display.py or separate health check module
@app.route('/health')
def health_check():
    return {'status': 'healthy', 'timestamp': time.time()}
```

Monitor:
- Container resource usage (CPU, memory)
- Redis connection status
- Processing latency metrics
- Error rates in logs

## Troubleshooting

Common deployment issues:

1. **Models not found**: Ensure model files are accessible in the container
2. **Redis connection failed**: Check REDIS_URL environment variable
3. **Port conflicts**: Ensure port 8501 is available
4. **Memory issues**: Increase container memory limits for ML models
5. **Slow startup**: Models loading can take 30-60 seconds on first start
