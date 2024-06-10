import os
from langchain_community.llms import OpenAI


davinch3 = OpenAI(model_name='text-davinci-003')
davinch3.predict('why python is the most popular language? answer in korean')
