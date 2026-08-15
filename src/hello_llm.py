import os,sys
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

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