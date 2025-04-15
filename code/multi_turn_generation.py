from langchain_community.llms import HuggingFaceTextGenInference
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from typing import List
import json
import os

os.environ["OPENAI_API_KEY"] = ""

llm = ChatOpenAI(model="gpt-4o-mini", temperature=1)

with open('dataset_v4.json') as f:
    d = json.load(f)

# Choose a sample and turn it into multi-turn dialogue
for i, elem in enumerate(d):

    prompt = "Take this email: " + elem['mail']
    prompt += "This mail was sent by: " + elem['sender']
    prompt += "\n\n Turn this mail into a multi-turn dialogue between the email sender and the assistant of management at Enron. Make it short and precise, the sender should introduce themself. The dialogue:"

    answer = llm.invoke(prompt).content

    d[i]['mail'] = answer

    print("[*] " + str(i + 1) + " of " + str(len(d)))
    
with open('dataset_multi_turn.json', 'w', encoding="utf-8") as f:
    json.dump(d, f, ensure_ascii=False, indent=4)
