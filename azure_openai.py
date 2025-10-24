import os
from dotenv import load_dotenv
from openai import AzureOpenAI

# Cargar variables de entorno
load_dotenv()

def test_azure_openai():
    """Prueba la conexión con Azure OpenAI"""
    
    print("=== PRUEBA DE AZURE OPENAI ===")
    
    # Verificar variables de entorno
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
    deployment_gpt4o = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")
    deployment_mini = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_MINI", "gpt-4o-mini")
    
    print(f"Endpoint: {endpoint}")
    print(f"API Key: {api_key[:10]}..." if api_key else "No encontrada")
    print(f"API Version: {api_version}")
    print(f"Deployment GPT-4o: {deployment_gpt4o}")
    print(f"Deployment GPT-4o-mini: {deployment_mini}")
    print()
    
    if not endpoint or not api_key:
        print("❌ ERROR: Faltan variables de entorno")
        return False
    
    try:
        # Crear cliente de Azure OpenAI
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version
        )
        
        print("✅ Cliente Azure OpenAI creado correctamente")
        
        # Probar con GPT-4o
        print(f"\n--- Probando {deployment_gpt4o} ---")
        try:
            response = client.chat.completions.create(
                model=deployment_gpt4o,
                messages=[
                    {"role": "user", "content": "Hola, ¿puedes responder con un simple 'Hola'?"}
                ],
                max_tokens=50
            )
            print(f"✅ {deployment_gpt4o} funcionando:")
            print(f"Respuesta: {response.choices[0].message.content}")
        except Exception as e:
            print(f"❌ Error con {deployment_gpt4o}: {e}")
        
        # Probar con GPT-4o-mini
        print(f"\n--- Probando {deployment_mini} ---")
        try:
            response = client.chat.completions.create(
                model=deployment_mini,
                messages=[
                    {"role": "user", "content": "Hola, ¿puedes responder con un simple 'Hola'?"}
                ],
                max_tokens=50
            )
            print(f"✅ {deployment_mini} funcionando:")
            print(f"Respuesta: {response.choices[0].message.content}")
        except Exception as e:
            print(f"❌ Error con {deployment_mini}: {e}")
            
    except Exception as e:
        print(f"❌ ERROR al crear cliente: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_azure_openai()