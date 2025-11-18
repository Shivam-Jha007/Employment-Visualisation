import google.generativeai as genai

# Paste your API Key here
genai.configure(api_key="AIzaSyCROvc4PCRdkiwQkKdtS1jYETLxkxEcZV0")

print("Querying Google for available models...")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print(f"Error connecting: {e}")