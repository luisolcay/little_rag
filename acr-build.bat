@echo off
REM Script optimizado para construir imagen con Azure Container Registry Build
REM Usa ACR Build para builds más rápidos y confiables

set ACR_NAME=orbecontaineregistry
set IMAGE_NAME=rag-platform
set IMAGE_TAG=latest

echo 🚀 Starting optimized Docker build with Azure Container Registry...
echo 📦 Building with ACR Build (optimized for Azure Container Apps)...
echo    Registry: %ACR_NAME%.azurecr.io
echo    Image: %IMAGE_NAME%:%IMAGE_TAG%
echo    Port: 8080 (Azure Container Apps standard)

REM Use ACR Build for better performance and reliability
az acr build ^
    --registry %ACR_NAME% ^
    --image %IMAGE_NAME%:%IMAGE_TAG% ^
    --image %IMAGE_NAME%:%DATE:~0,4%%DATE:~5,2%%DATE:~8,2%-%TIME:~0,2%%TIME:~3,2%%TIME:~6,2% ^
    --file Dockerfile ^
    --platform linux/amd64 ^
    .

if %ERRORLEVEL% NEQ 0 (
    echo ❌ Build failed
    exit /b 1
)

echo ✅ Build completed successfully!
echo 🎉 Image available at: %ACR_NAME%.azurecr.io/%IMAGE_NAME%:%IMAGE_TAG%
echo 📋 Next steps:
echo    1. Create Azure Container Apps Environment
echo    2. Create Container App pointing to this image
echo    3. Configure environment variables from .env file
echo    4. Set ingress to port 8080

