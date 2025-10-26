#!/bin/bash

# Script para construir y subir la imagen Docker a Azure Container Registry

set -e

echo "🚀 Starting Docker build and push process..."

# Variables
ACR_NAME="orbecontaineregistry"
IMAGE_NAME="rag-platform"
IMAGE_TAG="latest"
FULL_IMAGE_NAME="${ACR_NAME}.azurecr.io/${IMAGE_NAME}:${IMAGE_TAG}"

# Step 1: Build Docker image
echo "📦 Building Docker image..."
docker build -t ${FULL_IMAGE_NAME} .

echo "✅ Image built successfully"

# Step 2: Login to Azure Container Registry
echo "🔐 Logging in to Azure Container Registry..."
az acr login --name ${ACR_NAME}

echo "✅ Logged in successfully"

# Step 3: Push image to ACR
echo "⬆️  Pushing image to Azure Container Registry..."
docker push ${FULL_IMAGE_NAME}

echo "✅ Image pushed successfully"
echo "🎉 Done! Image available at: ${FULL_IMAGE_NAME}"

