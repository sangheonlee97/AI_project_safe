import os
from langchain.prompts import PromptTemplate, ChatPromptTemplate

#프롬프트 템플릿을 통해 매개변수 삽입 가능한 문자열로 변환
string_prompt = PromptTemplate.from_template("tell me a joke about {subject}")

#매개변수 삽입한 결과를 string_prompt_value에 할당
string_prompt_value = string_prompt.format_prompt(subject="soccer")

#채팅LLM이 아닌 LLM과 대화할 때 필요한 프롬프트 = string prompt
print(string_prompt_value)