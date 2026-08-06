import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

response = client.chat.completions.create(
    model="deepseek-v4-flash",
    max_tokens=500,
    messages=[{"role": "user", "content": "Πες μου ένα γεια από το ticket-triage project."}],
)

print(response.choices[0].message.content)