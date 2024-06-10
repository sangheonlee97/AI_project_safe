from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser
import os



# class CommaSeparatedListOutputParser(BaseOutputParser):
#     """LLM 아웃풋에 있는 ','를 분리해서 리턴하는 파서."""


#     def parse(self, text: str):
#         return text.strip().split(", ")

template = """
너는 5세 아이의 낱말놀이를 도와주는 AI야.
너는 공사장에서 객체의 상태를 파악하고 객체의 위치를 알려주는 AI야.
여러가지 객체의 이름과 좌표를 입력할테니, 내가 요구한 <특정 객체>가 탐지되면 좌표를 알려줘.
탐지된 <특정 객체>가 여러개라면, 콤마(,)로 구분해서 알려줘

특정 객체:"""
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

chain = LLMChain(
    llm=ChatOpenAI(),
    prompt=chat_prompt,
    # output_parser=CommaSeparatedListOutputParser()
)
chain.run("하이바미착용")