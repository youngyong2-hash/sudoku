import os
import streamlit as st
from PIL import Image
from google import genai
from google.genai import types

# 페이지 기본 설정
st.set_page_config(page_title="스도쿠 AI 도우미", page_icon="🧩", layout="centered")

st.title("🧩 스도쿠 AI Solver & Tutor")
st.write("스도쿠 이미지(손글씨 가능)를 업로드하면 힌트나 정답 검증을 진행합니다.")

# 기존 API 키 입력 부분을 아래와 같이 수정하세요

# 1. secrets.toml 파일이나 환경변수에서 키를 자동으로 읽어옵니다.
api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")

# 2. 만약 파일에 키가 없다면 사이드바에서 입력받도록 보완합니다.
if not api_key:
    api_key = st.sidebar.text_input("Gemini API Key를 입력하세요", type="password")
    if not api_key:
        st.info("👈 API Key를 설정해 주세요.")
        st.stop()

# Gemini 클라이언트 초기화 (Pro 모델 활용)
client = genai.Client(api_key=api_key)

# 이미지 업로드
uploaded_file = st.file_uploader("스도쿠 이미지 파일 업로드 (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="업로드된 스도쿠 이미지", use_container_width=True)

    # 서비스 선택
    option = st.radio("원하는 서비스를 선택하세요:", ["💡 단계별 힌트 받기", "✅ 정답 및 오류 검증하기"])

    if st.button("분석 실행", type="primary"):
        with st.spinner("Gemini Pro가 스도쿠 판을 정밀 분석 중입니다..."):
            
            if "힌트" in option:
                system_prompt = """
                당신은 친절하고 명확한 스도쿠 튜터입니다.
                업로드된 이미지의 스도쿠 판 상태(손글씨, 연필 메모 포함)를 정확히 인식하여:
                1. 확실하게 다음 숫자를 채울 수 있는 위치 1~2곳을 찾으세요.
                2. 해당 위치(몇 행 몇 열), 들어갈 숫자, 그리고 왜 들어가는지 논리적 이유를 단계별로 설명하세요.
                3. 전체 답을 스포일러하지 말고 힌트 형태로 작성하세요.
                """
                user_prompt = "현재 판 상태를 분석해서 다음에 채울 수 있는 숫자의 단계별 힌트를 알려주세요."
            else:
                system_prompt = """
                당신은 스도쿠 검증 시스템입니다.
                업로드된 이미지의 스도쿠 판을 확인하여 풀어낸 답안을 검증하세요:
                1. 1~9행, 1~9열, 9개의 3x3 영역에 숫자 중복이 있는지 전수 조사하세요.
                2. 오류가 있다면 정확한 위치(예: 8행 8열)와 중복된 숫자를 지적하세요.
                3. 틀린 부분이 없다면 완벽히 풀렸음을 확인해 주세요.
                """
                user_prompt = "이 스도쿠 판이 올바르게 풀렸는지 검증해 주세요."

            try:
                # Gemini Pro 모델 호출
                response = client.models.generate_content(
                    model="gemini-2.5-pro",
                    contents=[image, user_prompt],
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.1, # 논리적 추론을 위한 낮은 창의성 설정
                    ),
                )

                st.markdown("---")
                st.subheader("🤖 AI 분석 결과")
                st.markdown(response.text)

            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")