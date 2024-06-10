import os
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI

chatgpt = ChatOpenAI(model_name='gpt-3.5-turbo')
answer = chatgpt.predict('why python is the most popular language? answer in korean')
print(answer)