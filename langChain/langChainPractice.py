from langchain_community.chat_models import ChatOpenAI
llm = ChatOpenAI()

from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0,               # 창의성 (0.0 ~ 2.0) 
                 max_tokens=2048,             # 최대 토큰수
                 model_name='gpt-3.5-turbo',  # 모델명
                )

# 질의내용
question = '르세라핌 친일파야?'

# 질의
print(f'[답변]: {llm.predict(question)}')