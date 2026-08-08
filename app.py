import os
import json
import random
import streamlit as st
from PIL import Image, ImageDraw, ImageOps
from streamlit_cropper import st_cropper
from google import genai
from google.genai import types

# ------------------------------------------------------------------------------
# 1. 페이지 기본 설정 및 시크릿 키 로드
# ------------------------------------------------------------------------------
st.set_page_config(page_title="스도쿠 스마트 AI 도우미", page_icon="🧩", layout="centered")

# 모바일 화면 크기에 맞추는 CSS 강제 적용
st.markdown("""
    <style>
        .stApp { max-width: 100%; }
        iframe { max-width: 100% !important; width: 100% !important; }
        img { max-width: 100% !important; height: auto !important; }
    </style>
""", unsafe_allow_html=True)

api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")

if not api_key:
    api_key = st.sidebar.text_input("Gemini API Key를 입력하세요", type="password")
    if not api_key:
        st.info("👈 사이드바 또는 Secrets에 Gemini API Key를 설정해 주세요.")
        st.stop()

client = genai.Client(api_key=api_key)

# ------------------------------------------------------------------------------
# 2. 유틸리티 함수 (이미지 모바일 최적화, 스도쿠 알고리즘, 표 시각화)
# ------------------------------------------------------------------------------
PUZZLE_FILE = "puzzles_db.json"

def fit_to_mobile_screen(image, max_width=350):
    """스마트폰 한 화면에 사진 전체와 빨간 박스가 100% 들어오도록 크기를 자동 조절하는 함수"""
    image = ImageOps.exif_transpose(image) # 사진 방향 자동 교정
    w, h = image.size
    if w > max_width:
        scale = max_width / float(w)
        new_height = int(float(h) * float(scale))
        image = image.resize((max_width, new_height), Image.Resampling.LANCZOS)
    return image

def is_valid(board, row, col, num):
    for i in range(9):
        if board[row][i] == num or board[i][col] == num:
            return False
        if board[3 * (row // 3) + i // 3][3 * (col // 3) + i % 3] == num:
            return False
    return True

def solve_board(board):
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
    full_board = [[0] * 9 for _ in range(9)]
    solve_board(full_board)
    
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
    html = """
    <style>
        .sudoku-container { display: flex; justify-content: center; margin: 15px 0; }
        .sudoku-board { border-collapse: collapse; border: 3px solid #222222; background-color: #ffffff; }
        .sudoku-board td { width: 38px; height: 38px; text-align: center; vertical-align: middle; border: 1px solid #cccccc; font-size: 20px; font-weight: bold; color: #111111; }
        .sudoku-board td.solution-cell { color: #1d4ed8; background-color: #eff6ff; }
        .sudoku-board tr:nth-child(3n) td { border-bottom: 2px solid #222222; }
        .sudoku-board td:nth-child(3n) { border-right: 2px solid #222222; }
        .sudoku-board tr:first-child td { border-top: 2px solid #222222; }
        .sudoku-board td:first-child { border-left: 2px solid #222222; }
    </style>
    <div class="sudoku-container"><table class="sudoku-board">
    """
    for r in range(9):
        html += "<tr>"
        for c in range(9):
            val = puzzle[r][c]
            if val != 0:
                html += f"<td>{val}</td>"
            else:
                if solution and solution[r][c] != 0:
                    html += f'<td class="solution-cell">{solution[r][c]}</td>'
                else:
                    html += "<td></td>"
        html += "</tr>"
    html += "</table></div>"
    return html

def draw_errors_on_image(image, error_cells):
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
# TAB 1: 모바일 업로드, 100% 한눈에 들어오는 자르기, 자동 호환 AI 모델 분석
# ==============================================================================
with tab1:
    st.subheader("1. 스도쿠 이미지 가져오기")
    img_file = st.file_uploader("스도쿠 이미지를 촬영하거나 업로드하세요", type=["jpg", "jpeg", "png"])

    if img_file is not None:
        raw_image = Image.open(img_file)
        
        # 모바일 화면 폭(350px)에 맞추어 축소
        working_img = fit_to_mobile_screen(raw_image, max_width=350)

        # 회전 상태 관리
        if "rotate_angle" not in st.session_state:
            st.session_state["rotate_angle"] = 0

        st.subheader("2. 사진 방향 및 영역 맞추기")
        col_rot1, col_rot2 = st.columns(2)
        with col_rot1:
            if st.button("🔄 90° 회전"):
                st.session_state["rotate_angle"] = (st.session_state["rotate_angle"] - 90) % 360
        with col_rot2:
            if st.button("↩️ 방향 초기화"):
                st.session_state["rotate_angle"] = 0

        if st.session_state["rotate_angle"] != 0:
            working_img = working_img.rotate(st.session_state["rotate_angle"], expand=True)

        st.write("📱 **빨간 박스** 모서리를 조절해 9x9 영역에 맞춰주세요.")
        
        # 화면에 딱 들어오는 스크린 모드 적용
        cropped_img = st_cropper(
            working_img,
            realtime_update=True,
            box_color='#FF0000',
            aspect_ratio=(1, 1)
        )

        if cropped_img:
            st.image(cropped_img, caption="분석 영역 선택 완료", use_container_width=True)

            if st.button("💡 도움받기 (단 하나의 힌트 & 검증)", type="primary"):
                with st.spinner("AI가 스도쿠 판을 분석 중입니다..."):
                    system_prompt = """
                    당신은 엄격하고 명확한 스도쿠 검증 튜터입니다.
                    업로드된 이미지에서 9x9 스도쿠 판(인쇄체 및 손글씨)을 분석하세요.

                    반드시 아래 구조의 응답을 JSON 형식으로만 작성하세요:
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
                    1. errors: 손글씨 중 스도쿠 규칙에 위배되는(틀린) 숫자의 위치(row, col)를 기록하세요. 없으면 빈 배열 []을 반환하세요.
                    2. single_hint: 현재 확실히 바로 채울 수 있는 '단 한 칸'의 위치, 정답 숫자, 이유를 제시하세요.
                    """

                    # 구글 API 최신 지원 모델 후보군 (자동 대체 호출)
                    model_candidates = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
                    response = None
                    last_error = None

                    for model_name in model_candidates:
                        try:
                            response = client.models.generate_content(
                                model=model_name,
                                contents=[cropped_img, "이 스도쿠 판의 틀린 손글씨 위치와 바로 해결 가능한 단 하나의 힌트를 JSON으로 출력하세요."],
                                config=types.GenerateContentConfig(
                                    system_instruction=system_prompt,
                                    temperature=0.1,
                                    response_mime_type="application/json"
                                ),
                            )
                            if response and response.text:
                                break
                        except Exception as err:
                            last_error = err
                            continue

                    try:
                        if response and response.text:
                            result = json.loads(response.text)
                            errors = result.get("errors", [])
                            hint = result.get("single_hint", {})

                            st.markdown("---")
                            
                            if errors:
                                st.error(f"⚠️ **검증 결과:** 손글씨 중 틀린 부분이 {len(errors)}곳 발견되었습니다!")
                                annotated_image = draw_errors_on_image(cropped_img, errors)
                                st.image(annotated_image, caption="❌ 틀린 위치가 빨간색 X로 표시되었습니다", use_container_width=True)
                                
                                for err in errors:
                                    st.write(f"- **{err.get('row')}행 {err.get('col')}열**: {err.get('reason')}")
                            else:
                                st.success("✅ **검증 결과:** 현재 적힌 손글씨 중 규칙에 위배되는 숫자가 없습니다!")

                            if hint:
                                st.subheader("💡 바로 해결 가능한 한 칸 힌트")
                                st.info(
                                    f"👉 **위치:** **{hint.get('row')}행 {hint.get('col')}열**\n\n"
                                    f"👉 **정답 숫자:** **{hint.get('number')}**\n\n"
                                    f"👉 **풀이 이유:** {hint.get('reason')}"
                                )
                        else:
                            st.error(f"분석 중 오류가 발생했습니다: {last_error}")

                    except Exception as e:
                        st.error(f"결과 처리 중 오류가 발생했습니다: {e}")

# ==============================================================================
# TAB 2: 스도쿠 문제 만들기 & 저장된 데이터 보관함
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
        
        sol_saved = p_data.get("solution") or solve_sudoku_exact(pz_saved)
        show_saved_sol = st.toggle("🔍 저장된 문제 정답 보기", key="saved_sol_toggle")
        
        sol_param = sol_saved if show_saved_sol else None
        st.markdown(render_sudoku_board_html(pz_saved, solution=sol_param), unsafe_allow_html=True)
    else:
        st.info("아직 저장된 스도쿠 문제가 없습니다. 위에서 문제를 생성해 보세요!")
