import os
from dotenv import load_dotenv
from google import genai

print("Loading environment variables...")
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

result = client.models.embed_content(
    model="gemini-embedding-exp-03-07",
    contents=[
        "What is the meaning of life?",
        "What is your favorite color?",
    ],
    config={"output_dimensionality": 8},
)

print(result.embeddings)


for m in client.models.list():
    if "embedContent" in m.supported_actions:
        print(m.name)
