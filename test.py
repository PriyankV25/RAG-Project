# quickstart.py
import os
from mistralai.client import Mistral
from dotenv import load_dotenv

load_dotenv()
client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

response = client.chat.complete(
    model="mistral-large-latest",
    messages=[
        {"role": "user", "content": "What is Mistral AI?"}
    ],
)

print(response.choices[0].message.content)