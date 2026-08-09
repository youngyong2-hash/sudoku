import os
import json
import random
import datetime
import streamlit as st
from PIL import Image, ImageDraw, ImageOps
from streamlit_cropper import st_cropper
from google import genai
from google.genai import types

# ------------------------------------------------------------------------------
# 1. 페이지 기본 설정 및 모바일 CSS
# ------------------------------------------------------------------------------
st.set_page_config(page_title="스도쿠 AI 도우미", page_icon="🧩", layout="centered")

st.markdown("""
    <style>
        .stApp { max-width: 100%; padding-left: 0.5rem; padding-right: 0.5rem; }
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
# 2. 토큰 절감형 이미지 최적화 및 로컬 데이터 저장 함수
# ------------------------------------------------------------------------------
PUZZLE_FILE = "puzzles_db.json"

def compress_for_min_tokens(image, max_dim=320):
    """비전 토큰(Tile Tokens) 소모량을 극소화하기 위해 이미지를 320px 이하로 압축"""
    image = ImageOps.exif_transpose(image)
    w, h = image.size
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return image

def save_puzzle(difficulty, puzzle, solution):
    puzzles = load_puzzles()
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = {
        "id": len(puzzles) + 1,
        "difficulty": difficulty,
        "puzzle": puzzle,
        "solution": solution,
        "created_at": now_str
    }
    puzzles.append(new_entry)
    with open(PUZZLE_FILE, "w", encoding="utf-8") as f:
        json.dump(puzzles, f, ensure_ascii=False, indent=2)

def load_puzzles(difficulty=None):
    if os.path.exists(PUZZLE_FILE):
        try:
            with open(PUZZLE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if difficulty:
                    return [p for p in data if p.get("difficulty") == difficulty]
                return data
        except:
            return []
    return []

# ------------------------------------------------------------------------------
# 3. 알고리즘 및 표 시각화 함수
# ------------------------------------------------------------------------------
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

def render_sudoku_board_html(puzzle, solution=None):
    html = """
    <style>
        .sudoku-container { display: flex; justify-content: center; margin: 15px 0; }
        .sudoku-board { border-collapse: collapse; border: 3px solid #222222; background-color: #ffffff; }
        .sudoku-board td { width: 36px; height: 36px; text-align: center; vertical-align: middle; border: 1px solid #cccccc; font-size: 18px; font-weight: bold; color: #111111; }
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
# 4. 메인 UI (탭 구조)
# ------------------------------------------------------------------------------
st.title("🧩 스도쿠 AI 스마트 도우미")

tab1, tab2 = st.tabs(["📸 이미지 업로드 & 도움받기", "🎲 문제 만들기 & 보관함"])

# ==============================================================================
# TAB 1: 토큰 초절감형 이미지 업로드 & 도움받기
# ==============================================================================
with tab1:
    st.subheader("1. 스도쿠 이미지 가져오기")
    img_file = st.file_uploader("스도쿠 이미지를 촬영하거나 업로드하세요", type=["jpg", "jpeg", "png"])

    if img_file is not None:
        raw_image = Image.open(img_file)
        # 이미지 크기를 압축하여 비전 토큰 소모 최소화
        working_img = compress_for_min_tokens(raw_image, max_dim=320)

        if "rotate_angle" not in st.session_state:
            st.session_state["rotate_angle"] = 0

        st.subheader("2. 사진 방향 및 영역 설정")
        col_rot1, col_rot2 = st.columns(2)
        with col_rot1:
            if st.button("🔄 90° 회전"):
                st.session_state["rotate_angle"] = (st.session_state["rotate_angle"] - 90) % 360
        with col_rot2:
            if st.button("↩️ 방향 초기화"):
                st.session_state["rotate_angle"] = 0

        if st.session_state["rotate_angle"] != 0:
            working_img = working_img.rotate(st.session_state["rotate_angle"], expand=True)

        use_cropper = st.checkbox("✂️ 빨간 박스로 9x9 잘라내기 사용", value=True)

        target_img = working_img
        if use_cropper:
            st.write("📱 모서리를 조절해 9x9 영역에 맞추세요.")
            target_img = st_cropper(
                working_img,
                realtime_update=True,
                box_color='#FF0000',
                aspect_ratio=(1, 1)
            )

        if target_img:
            st.image(target_img, caption="분석 영역 선택 완료", use_container_width=True)

            if st.button("💡 도움받기 (단 하나의 힌트 & 검증)", type="primary"):
                with st.spinner("최소 토큰 모드로 정밀 분석 중..."):
                    # 토큰 절약을 위한 최소 프롬프트 구조
                    system_prompt = """Analyze 9x9 Sudoku image. Output ONLY valid JSON:
                    {
                        "errors": [{"row": int(1-9), "col": int(1-9), "reason": "str"}],
                        "single_hint": {"row": int(1-9), "col": int(1-9), "number": int(1-9), "reason": "str"}
                    }"""

                    try:
                        # 초저가/고속 모델 단일 지정 (토큰 비용 최소화)
                        response = client.models.generate_content(
                            model="gemini-2.5-flash-lite",
                            contents=[target_img, "Find errors and 1 exact hint."],
                            config=types.GenerateContentConfig(
                                system_instruction=system_prompt,
                                temperature=0.1,
                                response_mime_type="application/json"
                            ),
                        )

                        if response and response.text:
                            result = json.loads(response.text)
                            errors = result.get("errors", [])
                            hint = result.get("single_hint", {})

                            st.markdown("---")
                            
                            if errors:
                                st.error(f"⚠️ **검증 결과:** 손글씨 중 틀린 부분이 {len(errors)}곳 발견되었습니다!")
                                annotated_image = draw_errors_on_image(target_img, errors)
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
                            st.error("분석 응답을 불러오지 못했습니다. 다시 시도해 주세요.")

                    except Exception as e:
                        st.error(f"분석 중 오류 발생: {e}")

# ==============================================================================
# TAB 2: 스도쿠 문제 만들기 & 보관함 (100% 로컬 계산 - 토큰 0소모)
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
        st.success(f"새로운 {difficulty} 문제가 생성되어 보관함에 저장되었습니다!")

    if "current_puzzle" in st.session_state:
        st.write(f"### 📋 생성된 문제 ({st.session_state['current_diff']})")
        show_sol = st.toggle("🔍 정답 보기 (파란색 빈칸 채우기)", key="gen_sol_toggle")
        
        pz = st.session_state["current_puzzle"]
        sol = st.session_state["current_solution"] if show_sol else None
        
        st.markdown(render_sudoku_board_html(pz, solution=sol), unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📁 저장된 문제 보관함")
    
    filter_diff = st.radio("조회할 난이도 선택", ["전체", "초급", "중급", "고급"], horizontal=True)
    target_diff = None if filter_diff == "전체" else filter_diff
    
    saved_puzzles = load_puzzles(target_diff)

    if saved_puzzles:
        st.write(f"총 **{len(saved_puzzles)}개**의 저장된 문제가 있습니다.")
        selected_idx = st.selectbox(
            "불러올 문제를 선택하세요",
            options=list(range(len(saved_puzzles))),
            format_func=lambda i: f"#{saved_puzzles[i]['id']} [{saved_puzzles[i]['difficulty']}] ({saved_puzzles[i].get('created_at', '')})"
        )

        p_data = saved_puzzles[selected_idx]
        pz_saved = p_data["puzzle"]
        sol_saved = p_data.get("solution") or solve_sudoku_exact(pz_saved)
        
        show_saved_sol = st.toggle("🔍 저장된 문제 정답 보기", key="saved_sol_toggle")
        
        sol_param = sol_saved if show_saved_sol else None
        st.markdown(render_sudoku_board_html(pz_saved, solution=sol_param), unsafe_allow_html=True)
    else:
        st.info("선택한 난이도에 저장된 스도쿠 문제가 없습니다. 위에서 새 문제를 만들어 보세요!")
