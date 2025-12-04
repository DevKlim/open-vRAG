import google.generativeai as genai
import os

API_KEY = "AIzaSyBJfMhWQFUI1Dg3nu3i1WFEWUy-v10nwyc"
genai.configure(api_key=API_KEY)

print("Listing available models:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)
