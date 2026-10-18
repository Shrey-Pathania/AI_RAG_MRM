import os,sys
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

# client = OpenAI()
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)


# def ask(question: str) -> str :
def ask(question):
    resp = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages=[ 
            {"role": "system", "content": "You are concise."}, 
            {"role": "user", "content": question}, ], 
        temperature=0.3, 
    ) 
    return resp.choices[0].message.content

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Say hello."
    print(ask(q))