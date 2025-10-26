#!/bin/bash

# Script optimizado para construir y subir imagen a Azure Container Registry
# Usa ACR Build para builds más rápidos y confiables

set -e

echo "🚀 Starting optimized Docker build with Azure Container Registry..."

# Variables
ACR_NAME="orbecontaineregistry"
IMAGE_NAME="rag-platform"
IMAGE_TAG="latest"

echo "📦 Building with ACR Build (optimized for Azure Container Apps)..."
echo "   Registry: ${ACR_NAME}.azurecr.io"
echo "   Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "   Port: 8080 (Azure Container Apps standard)"

# Use ACR Build for better performance and reliability
az acr build \
    --registry ${ACR_NAME} \
    --image ${IMAGE_NAME}:${IMAGE_TAG} \
    --image ${IMAGE_NAME}:$(date +%Y%m%d-%H%M%S) \
    --file Dockerfile \
    --platform linux/amd64 \
    .

echo "✅ Build completed successfully!"
echo "🎉 Image available at: ${ACR_NAME}.azurecr.io/${IMAGE_NAME}:${IMAGE_TAG}"
echo "📋 Next steps:"
echo "   1. Create Azure Container Apps Environment"
echo "   2. Create Container App pointing to this image"
echo "   3. Configure environment variables from .env file"
echo "   4. Set ingress to port 8080"

