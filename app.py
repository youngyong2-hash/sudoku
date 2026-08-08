import os
import json
import random
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_cropper import st_cropper
from google import genai
from google.genai import types

# ------------------------------------------------------------------------------
# 1. 페이지 기본 설정 및 시크릿 키 로드
# ------------------------------------------------------------------------------
st.set_page_config(page_title="스도쿠 스마트 AI 도우미", page_icon="🧩", layout="centered")

api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")

if not api_key:
    api_key = st.sidebar.text_input("Gemini API Key를 입력하세요", type="password")
    if not api_key:
        st.info("👈 사이드바 또는 Secrets에 Gemini API Key를 설정해 주세요.")
        st.stop()

client = genai.Client(api_key=api_key)

# ------------------------------------------------------------------------------
# 2. 유틸리티 함수 (스도쿠 생성기, 정답 계산기, 데이터 저장, 9x9 표 시각화)
# ------------------------------------------------------------------------------
PUZZLE_FILE = "puzzles_db.json"

def is_valid(board, row, col, num):
    for i in range(9):
        if board[row][i] == num or board[i][col] == num:
            return False
        if board[3 * (row // 3) + i // 3][3 * (col // 3) + i % 3] == num:
            return False
    return True

def solve_board(board):
    """랜덤 스도쿠 판 생성용 백트래킹 풀이 함수"""
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                nums = list(range(1, 10))
                random.shuffle(nums)
                for num in nums:
                    if is_valid(board, row, col, num):
                        board[row][col] = num
                        if solve_board(board):
                            return True
                        board[row][col] = 0
                return False
    return True

def solve_sudoku_exact(board):
    """지정된 스도쿠 판의 정확한 정답 판을 계산해주는 알고리즘"""
    board_copy = [row[:] for row in board]
    def solve(b):
        for row in range(9):
            for col in range(9):
                if b[row][col] == 0:
                    for num in range(1, 10):
                        if is_valid(b, row, col, num):
                            b[row][col] = num
                            if solve(b):
                                return True
                            b[row][col] = 0
                    return False
        return True
    solve(board_copy)
    return board_copy

def generate_sudoku_puzzle(difficulty):
    """난이도별 스도쿠 문제 및 전체 정답판 생성"""
    full_board = [[0] * 9 for _ in range(9)]
    solve_board(full_board)  # 정답판 완성
    
    clues_count = {'초급': 38, '중급': 30, '고급': 24}.get(difficulty, 30)
    remove_count = 81 - clues_count
    
    puzzle = [row[:] for row in full_board]
    positions = [(r, c) for r in range(9) for c in range(9)]
    random.shuffle(positions)
    
    for i in range(remove_count):
        r, c = positions[i]
        puzzle[r][c] = 0
        
    return puzzle, full_board

def save_puzzle(difficulty, puzzle, solution):
    puzzles = load_puzzles()
    new_entry = {
        "id": len(puzzles) + 1,
        "difficulty": difficulty,
        "puzzle": puzzle,
        "solution": solution
    }
    puzzles.append(new_entry)
    with open(PUZZLE_FILE, "w", encoding="utf-8") as f:
        json.dump(puzzles, f, ensure_ascii=False, indent=2)

def load_puzzles():
    if os.path.exists(PUZZLE_FILE):
        try:
            with open(PUZZLE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []

def render_sudoku_board_html(puzzle, solution=None):
    """
    9x9 스도쿠 판을 굵은 3x3 경계선이 있는 HTML 표로 렌더링.
    solution이 제공되면 원래 빈칸(0) 자리에 정답 숫자를 파란색으로 표시합니다.
    """
    html = """
    <style>
        .sudoku-container {
            display: flex;
            justify-content: center;
            margin: 15px 0;
        }
        .sudoku-board {
            border-collapse: collapse;
            border: 3px solid #222222;
            background-color: #ffffff;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .sudoku-board td {
            width: 38px;
            height: 38px;
            text-align: center;
            vertical-align: middle;
            border: 1px solid #cccccc;
            font-size: 20px;
            font-weight: bold;
            color: #111111;
        }
        /* 정답 숫자 전용 스타일 (파란색 글씨 + 연한 파란 배경) */
        .sudoku-board td.solution-cell {
            color: #1d4ed8;
            background-color: #eff6ff;
        }
        /* 3x3 구역 구분선 굵게 지정 */
        .sudoku-board tr:nth-child(3n) td {
            border-bottom: 2px solid #222222;
        }
        .sudoku-board td:nth-child(3n) {
            border-right: 2px solid #222222;
        }
        .sudoku-board tr:first-child td {
            border-top: 2px solid #222222;
        }
        .sudoku-board td:first-child {
            border-left: 2px solid #222222;
        }
    </style>
    <div class="sudoku-container">
    <table class="sudoku-board">
    """
    for r in range(9):
        html += "<tr>"
        for c in range(9):
            val = puzzle[r][c]
            if val != 0:
                html += f"<td>{val}</td>"
            else:
                # 빈칸일 때 solution이 온 경우 정답 숫자를 파란색 스타일 적용
                if solution and solution[r][c] != 0:
                    sol_val = solution[r][c]
                    html += f'<td class="solution-cell">{sol_val}</td>'
                else:
                    html += "<td></td>"
        html += "</tr>"
    html += "</table></div>"
    return html

def draw_errors_on_image(image, error_cells):
    """틀린 손글씨 위치(행, 열)에 빨간색 X 표시를 그리는 함수"""
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    w, h = annotated.size
    cell_w = w / 9.0
    cell_h = h / 9.0

    for item in error_cells:
        r, c = item.get("row", 0), item.get("col", 0)
        if 1 <= r <= 9 and 1 <= c <= 9:
            row_idx, col_idx = r - 1, c - 1
            x1 = col_idx * cell_w + cell_w * 0.15
            y1 = row_idx * cell_h + cell_h * 0.15
            x2 = (col_idx + 1) * cell_w - cell_w * 0.15
            y2 = (row_idx + 1) * cell_h - cell_h * 0.15

            stroke_width = max(3, int(w / 80))
            draw.line([(x1, y1), (x2, y2)], fill="red", width=stroke_width)
            draw.line([(x1, y2), (x2, y1)], fill="red", width=stroke_width)

    return annotated

# ------------------------------------------------------------------------------
# 3. 메인 UI (탭 구조)
# ------------------------------------------------------------------------------
st.title("🧩 스도쿠 AI 스마트 도우미")

tab1, tab2 = st.tabs(["📸 이미지 업로드 & 도움받기", "🎲 문제 만들기 & 보관함"])

# ==============================================================================
# TAB 1: 이미지 업로드(촬영/앨범 겸용), 자르기, 딱 한 칸 힌트 및 X 표시
# ==============================================================================
with tab1:
    st.subheader("1. 스도쿠 이미지 가져오기")
    img_file = st.file_uploader("스도쿠 이미지를 촬영하거나 업로드하세요", type=["jpg", "jpeg", "png"])

    if img_file is not None:
        raw_image = Image.open(img_file)
        
        st.subheader("2. 9x9 영역 잘라내기")
        st.write("격자 모서리를 조절하여 스도쿠 9x9 테두리에 딱 맞게 맞춰주세요.")
        
        cropped_img = st_cropper(
            raw_image,
            realtime_update=True,
            box_color='#FF0000',
            aspect_ratio=(1, 1)
        )

        if cropped_img:
            st.image(cropped_img, caption="자른 스도쿠 영역", use_container_width=True)

            if st.button("💡 도움받기 (단 하나의 힌트 & 검증)", type="primary"):
                with st.spinner("AI가 스도쿠 판을 정밀 검석 중입니다..."):
                    system_prompt = """
                    당신은 엄격하고 명확한 스도쿠 검증 튜터입니다.
                    업로드된 9x9 스도쿠 이미지(인쇄체 숫자와 손글씨 숫자 포함)를 분석하여 반환하세요.

                    반드시 아래 구조의 응답을 JSON 형식으로만 작성하세요 (다른 일반 텍스트 제외):
                    {
                        "errors": [
                            {"row": 행번호(1-9), "col": 열번호(1-9), "reason": "오류 이유"}
                        ],
                        "single_hint": {
                            "row": 행번호(1-9),
                            "col": 열번호(1-9),
                            "number": 들어갈숫자(1-9),
                            "reason": "해당 칸에 이 숫자가 들어가는 논리적 이유"
                        }
                    }

                    [주의 규칙]
                    1. errors: 손글씨로 적혀있는 숫자 중 스도쿠 규칙에 위배되는(틀린) 숫자의 정확한 행, 열 위치를 기록하세요. 틀린 숫자가 없으면 빈 배열 []을 반환하세요.
                    2. single_hint: 현재 상황에서 확실하게 바로 채울 수 있는 '딱 한 칸'의 위치와 정답 숫자, 이유를 제시하세요.
                    """

                    try:
                        response = client.models.generate_content(
                            model="gemini-2.5-pro",
                            contents=[cropped_img, "이 스도쿠 판의 틀린 손글씨 위치와 바로 해결 가능한 단 하나의 힌트를 JSON으로 출력하세요."],
                            config=types.GenerateContentConfig(
                                system_instruction=system_prompt,
                                temperature=0.1,
                                response_mime_type="application/json"
                            ),
                        )

                        result = json.loads(response.text)
                        errors = result.get("errors", [])
                        hint = result.get("single_hint", {})

                        st.markdown("---")
                        
                        # 1. 틀린 숫자가 있는 경우 Red X 표시
                        if errors:
                            st.error(f"⚠️ **검증 결과:** 손글씨 중 틀린 부분이 {len(errors)}곳 발견되었습니다!")
                            annotated_image = draw_errors_on_image(cropped_img, errors)
                            st.image(annotated_image, caption="❌ 틀린 위치가 빨간색 X로 표시되었습니다", use_container_width=True)
                            
                            for err in errors:
                                st.write(f"- **{err.get('row')}행 {err.get('col')}열**: {err.get('reason')}")
                        else:
                            st.success("✅ **검증 결과:** 현재 적힌 손글씨 중 규칙에 위배되는 숫자가 없습니다!")

                        # 2. 딱 한 칸 해결 힌트 출력
                        if hint:
                            st.subheader("💡 바로 해결 가능한 한 칸 힌트")
                            st.info(
                                f"👉 **위치:** **{hint.get('row')}행 {hint.get('col')}열**\n\n"
                                f"👉 **정답 숫자:** **{hint.get('number')}**\n\n"
                                f"👉 **풀이 이유:** {hint.get('reason')}"
                            )

                    except Exception as e:
                        st.error(f"분석 중 오류가 발생했습니다: {e}")

# ==============================================================================
# TAB 2: 스도쿠 문제 만들기 & 저장된 데이터 보관함 (정답 보기 옵션 추가)
# ==============================================================================
with tab2:
    st.subheader("🎲 난이도별 스도쿠 문제 생성")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        difficulty = st.selectbox("난이도를 선택하세요", ["초급", "중급", "고급"])
    with col2:
        st.write("")
        st.write("")
        gen_btn = st.button("문제 생성", type="primary")

    if gen_btn:
        new_puzzle, new_solution = generate_sudoku_puzzle(difficulty)
        save_puzzle(difficulty, new_puzzle, new_solution)
        st.session_state["current_puzzle"] = new_puzzle
        st.session_state["current_solution"] = new_solution
        st.session_state["current_diff"] = difficulty
        st.success(f"새로운 {difficulty} 문제가 생성되고 보관함에 저장되었습니다!")

    if "current_puzzle" in st.session_state:
        st.write(f"### 📋 생성된 문제 ({st.session_state['current_diff']})")
        
        # 정답 보기 토글 스위치
        show_sol = st.toggle("🔍 정답 보기 (파란색 빈칸 채우기)", key="gen_sol_toggle")
        
        pz = st.session_state["current_puzzle"]
        sol = st.session_state["current_solution"] if show_sol else None
        
        st.markdown(render_sudoku_board_html(pz, solution=sol), unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📁 저장된 문제 보관함")
    saved_puzzles = load_puzzles()

    if saved_puzzles:
        st.write(f"총 **{len(saved_puzzles)}개**의 문제 데이터가 저장되어 있습니다.")
        selected_id = st.selectbox(
            "불러올 문제를 선택하세요",
            options=[p["id"] for p in saved_puzzles],
            format_func=lambda x: f"문제 ID #{x} ({next(p['difficulty'] for p in saved_puzzles if p['id'] == x)})"
        )

        p_data = next(p for p in saved_puzzles if p["id"] == selected_id)
        pz_saved = p_data["puzzle"]
        
        # 이전 저장 파일에 solution이 없더라도 백트래킹 알고리즘으로 자동 풀이
        sol_saved = p_data.get("solution") or solve_sudoku_exact(pz_saved)
        
        # 저장된 문제용 정답 보기 토글 스위치
        show_saved_sol = st.toggle("🔍 저장된 문제 정답 보기", key="saved_sol_toggle")
        
        sol_param = sol_saved if show_saved_sol else None
        st.markdown(render_sudoku_board_html(pz_saved, solution=sol_param), unsafe_allow_html=True)
    else:
        st.info("아직 저장된 스도쿠 문제가 없습니다. 위에서 문제를 생성해 보세요!")
