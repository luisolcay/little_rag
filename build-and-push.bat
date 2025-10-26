@echo off
REM Script para construir y subir la imagen Docker a Azure Container Registry

set ACR_NAME=orbecontaineregistry
set IMAGE_NAME=rag-platform
set IMAGE_TAG=latest
set FULL_IMAGE_NAME=%ACR_NAME%.azurecr.io/%IMAGE_NAME%:%IMAGE_TAG%

echo 🚀 Starting Docker build and push process...
echo 📦 Building optimized image for Azure Container Apps...

REM Step 1: Build Docker image
echo 📦 Building Docker image...
docker build -t %FULL_IMAGE_NAME% .

if %ERRORLEVEL% NEQ 0 (
    echo ❌ Build failed
    exit /b 1
)

echo ✅ Image built successfully

REM Step 2: Login to Azure Container Registry
echo 🔐 Logging in to Azure Container Registry...
az acr login --name %ACR_NAME%

if %ERRORLEVEL% NEQ 0 (
    echo ❌ Login failed
    exit /b 1
)

echo ✅ Logged in successfully

REM Step 3: Push image to ACR
echo ⬆️  Pushing image to Azure Container Registry...
docker push %FULL_IMAGE_NAME%

if %ERRORLEVEL% NEQ 0 (
    echo ❌ Push failed
    exit /b 1
)

echo ✅ Image pushed successfully
echo 🎉 Done! Image available at: %FULL_IMAGE_NAME%

