import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import tempfile
import os

# YOLO 클래스 이름을 인덱스에 매핑
class_names = [
    '소화기', '화재위험', '보행금지', '미지정위험', '생물학적위험', '고양이', '건설차량', '부식위험', '개', '음료금지', 
    '귀덮개', '감전위험', '폭발위험', '응급처치', '지게차', '지게차위험', '안경', '장갑', '경비견', '하네스', '하이바', 
    '레이저빔', '마스크', '불꽃', '굴착금지', '귀덮개미착용', '안경미착용', '장갑미착용', '하이바미착용', '마스크미착용', 
    '신발미착용', '흡연금지', '조끼미착용', '비전리방사선', '문열림위험', '작업자', '전화사용', '방사선위험', '신발', 
    '독성위험', '안전조끼', '안경착용위험', '호흡기착용'
]

class_dict = {name: idx for idx, name in enumerate(class_names)}

class YOLODetectionTool:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        # self.font = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", 24)
        np.random.seed(0)
        self.colors = np.random.randint(0, 255, size=(len(class_names), 3), dtype=np.uint8)
        self.class_id = None

    def set_class_name(self, class_name):
        if class_name in class_dict:
            self.class_id = class_dict[class_name]
        else:
            raise ValueError(f"클래스 이름 '{class_name}'에 해당하는 ID를 찾을 수 없습니다.")

    def run(self, frame):
        if self.class_id is None:
            raise ValueError("탐지할 클래스가 설정되지 않았습니다.")
        
        results = self.model(frame)
        predictions = results[0].boxes

        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)

        for pred in predictions:
            cls = int(pred.cls[0])
            if cls == self.class_id:
                x1, y1, x2, y2 = map(int, pred.xyxy[0])
                conf = pred.conf[0]
                label = f"{class_names[cls]} {conf:.2f}"
                color = self.colors[cls].tolist()
                draw.rectangle(((x1, y1), (x2, y2)), outline=tuple(color), width=2)
                draw.text((x1, y1 - 50), label, fill=tuple(color) + (255,))

        return cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

import openai
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI
import os

# OpenAI API 키 설정 (환경 변수나 직접 입력)

# 프롬프트 템플릿 정의
template = """
    너는 욜로 모델의 라벨을 맞추는 모델이야. 문장을 입력받으면 아래의 라벨들 중에서 그 문장이 설명하는 라벨을 한개만 정해줘. 라벨이기 때문에 너가 말하는 결과에 따옴표나 컴마(,), 점(.)이 포함되면 안돼.
    소화기, 화재위험, 보행금지, 미지정위험, 생물학적위험, 고양이, 건설차량, 부식위험, 개, 음료금지, 
    귀덮개, 감전위험, 폭발위험, 응급처치, 지게차, 지게차위험, 안경, 장갑, 경비견, 하네스, 하이바, 
    레이저빔, 마스크, 불꽃, 굴착금지, 귀덮개미착용, 안경미착용, 장갑미착용, 하이바미착용, 마스크미착용, 
    신발미착용, 흡연금지, 조끼미착용, 비전리방사선, 문열림위험, 작업자, 전화사용, 방사선위험, 신발, 
    독성위험, 안전조끼, 안경착용위험, 호흡기착용
    
    description:{description}"""
prompt = PromptTemplate(template=template, input_variables=["description"])

# LLM 구성 (여기서는 OpenAI GPT-3.5-turbo 모델을 사용)
llm = OpenAI(model_name="gpt-3.5-turbo")

# YOLO Tool 구성
yolo_tool = YOLODetectionTool(model_path="best.pt")

class CustomLLMChain:
    def __init__(self, llm, prompt, yolo_tool):
        self.llm_chain = LLMChain(llm=llm, prompt=prompt)
        self.yolo_tool = yolo_tool
        self.current_description = ""
        self.current_class_name = ""

    def __call__(self, inputs):
        description = inputs['description']
        frame = inputs['frame']

        # 입력된 설명이 변경되었을 때만 LLM을 호출하여 클래스 이름 추출
        if description != self.current_description:
            response = self.llm_chain.run({"description": description})
            class_name = response.strip()
            self.current_class_name = class_name
            self.current_description = description

            # YOLO Tool에 클래스 이름 설정
            self.yolo_tool.set_class_name(self.current_class_name)

        # YOLO Tool을 사용하여 객체 탐지 및 주석 추가
        annotated_frame = self.yolo_tool.run(frame)
        return {"annotated_frame": annotated_frame}

# Streamlit 앱 설정
st.title("YOLO Video Annotation with Streamlit")

# 비디오 파일 업로드
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

# 사용자 입력
user_input = st.text_input("Enter the object description (e.g., '조끼 안 입은 사람 찾아줘')")

if st.button("Process Video"):
    if uploaded_file is not None and user_input:
        # 비디오 파일 저장
        input_video_path = "input_video.mp4"
        with open(input_video_path, "wb") as f:
            f.write(uploaded_file.read())

        # 비디오 캡처 객체 생성
        cap = cv2.VideoCapture(input_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # LangChain을 통해 YOLO Tool을 호출하여 클래스 이름 설정
        chain = CustomLLMChain(llm=llm, prompt=prompt, yolo_tool=yolo_tool)
        chain({"description": user_input, "frame": np.zeros((height, width, 3), dtype=np.uint8)})

        # 실시간 비디오 스트리밍
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 프레임이 비어 있지 않은지 확인
            if frame is None or frame.size == 0:
                continue

            # YOLO Tool을 호출하여 객체 탐지 및 주석 추가
            result = chain({"description": user_input, "frame": frame})
            annotated_frame = result["annotated_frame"]

            # 프레임을 Streamlit에 디스플레이
            stframe.image(annotated_frame, channels="BGR")

        cap.release()
