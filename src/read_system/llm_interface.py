# llm_interface.py

from openai import OpenAI
from typing import List, Dict, Tuple
from dotenv import load_dotenv
import os

load_dotenv()

    
    
class Client:
    def __init__(self, model: str = "local-4090-gpt-oss", max_context_tokens: int = 8000):
        self.model = model
        self.max_tokens = max_context_tokens
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE_URL"))

    def truncate_history(self, history: List[Dict[str, str]], system_prompt: str, max_total_tokens: int = 7000):
        est_token_per_msg = 150
        budget = (max_total_tokens - len(system_prompt) // 4) // est_token_per_msg
        return history[-budget:]

    def chat(self, system_prompt: str, user_query: str, history: List[Dict[str, str]]=[]) -> Tuple[str, List[Dict[str, str]]]:
        context = self.truncate_history(history, system_prompt)
        messages = [{"role": "system", "content": system_prompt}] + context
        messages.append({"role": "user", "content": user_query})
        # import json
        # response = self.client.responses.create(
        #     model=self.model,
        #     input=[{"role": "user", "content": json.dumps(messages)}],
        #     reasoning={"effort": "medium"},
        #     temperature=0,
        # )
        # print(f"Response: {response}")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=1.0,
        )
        reply = response.choices[0].message.content

        history.append({"role": "user", "content": user_query})
        history.append({"role": "assistant", "content": reply})
        return reply, history

if __name__ == "__main__":
    gpt = Client()
    system_prompt = "You are a helpful assistant. Use only provided context to answer."
    
    while True:
        q = input("You: ")
        if q.lower() in ("exit", "quit"):
            break
        print("Bot:", gpt.chat(system_prompt, q))