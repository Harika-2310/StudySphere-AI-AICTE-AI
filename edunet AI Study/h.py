import google.generativeai as genai

genai.configure(api_key="AIzaSyBM8yC1fOhrD4HeVNTbJ_iXYUsc_7CIYAY")
for m in genai.list_models():
    print(m.name)