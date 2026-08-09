import os
import json
import random
import hashlib
import datetime
import io  # PDF 데이터를 메모리에 담기 위해 필요
import streamlit as st
from PIL import Image, ImageDraw, ImageOps
from streamlit_cropper import st_cropper
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from fpdf import FPDF  # PDF 생성을 위한 라이브러리

# ------------------------------------------------------------------------------
# 1. 페이지 기본 설정 및 모바일 UI CSS
# ------------------------------------------------------------------------------
st.set_page_config(page_title="스도쿠 AI 도우미", page_icon="🧩", layout="centered")

st.markdown("""
    <style>
        .stApp { max-width: 100%; padding-left: 0.5rem; padding-right: 0.5rem; }
        iframe { max-width: 100% !important; width: 100% !important; }
        img { max-width: 100% !important; height: auto !important; }
    </style>
""", unsafe_allow_html=True)

# Gemini API 키 설정 (Secrets 사용 추천)
api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")

if not api_key:
    api_key = st.sidebar.text_input("Gemini API Key를 입력하세요", type="password")
    if not api_key:
        st.info("👈 사이드바 또는 Streamlit Cloud Settings ➔ Secrets에 Gemini API Key를 설정해 주세요.")
        st.stop()


@st.cache_resource(show_spinner=False)
def get_client(key: str):
    """API 키가 바뀌지 않는 한 client를 재사용해 불필요한 재생성을 방지"""
    return genai.Client(api_key=key)


client = get_client(api_key)

# ------------------------------------------------------------------------------
# 1-1. Gemini 응답 스키마 정의 (구조화된 출력 강제)
# ------------------------------------------------------------------------------
class SudokuError(BaseModel):
    row: int = Field(..., ge=1, le=9)
    col: int = Field(..., ge=1, le=9)
    reason: str


class SudokuHint(BaseModel):
    row: int = Field(..., ge=1, le=9)
    col: int = Field(..., ge=1, le=9)
    number: int = Field(..., ge=1, le=9)
    reason: str


class SudokuAnalysis(BaseModel):
    errors: list[SudokuError] = Field(default_factory=list)
    single_hint: SudokuHint | None = None


# ------------------------------------------------------------------------------
# 2. 이미지 처리 및 데이터 저장 함수
# ------------------------------------------------------------------------------
PUZZLE_FILE = "puzzles_db.json"


def compress_for_display(image, max_dim=350):
    """모바일 화면 표시용 저해상도 리사이즈"""
    image = ImageOps.exif_transpose(image)
    w, h = image.size
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return image


def prepare_for_api(image, max_dim=768):
    """AI 인식 정확도를 위해 화면 표시본보다 더 높은 해상도로 별도 준비"""
    image = ImageOps.exif_transpose(image)
    w, h = image.size
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return image


def image_hash(image: Image.Image) -> str:
    """PIL 이미지는 기본 동등비교(==)가 객체 identity 기준이라 매 rerun마다
    다른 객체로 취급되는 문제가 있어, 픽셀 데이터 해시로 실제 변경 여부를 판단"""
    return hashlib.md5(image.tobytes()).hexdigest()


def load_puzzles(difficulty=None):
    if os.path.exists(PUZZLE_FILE):
        try:
            with open(PUZZLE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if difficulty:
                    return [p for p in data if p.get("difficulty") == difficulty]
                return data
        except (json.JSONDecodeError, ValueError):
            return []
        except Exception:
            return []
    return []


def save_puzzle(difficulty, puzzle, solution):
    puzzles = load_puzzles()
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # 삭제/동시접속 시 id 중복을 막기 위해 max(id)+1 사용
    next_id = (max((p.get("id", 0) for p in puzzles), default=0)) + 1
    new_entry = {
        "id": next_id,
        "difficulty": difficulty,
        "puzzle": puzzle,
        "solution": solution,
        "created_at": now_str,
    }
    puzzles.append(new_entry)
    try:
        with open(PUZZLE_FILE, "w", encoding="utf-8") as f:
            json.dump(puzzles, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"데이터 저장 중 오류 발생: {e}")


# ------------------------------------------------------------------------------
# 3. 핵심 알고리즘 (생성/검증/시각화/PDF)
# ------------------------------------------------------------------------------
def is_valid(board, row, col, num):
    for i in range(9):
        if board[row][i] == num or board[i][col] == num:
            return False
        if board[3 * (row // 3) + i // 3][3 * (col // 3) + i % 3] == num:
            return False
    return True


def solve_board(board):
    """무작위 순서로 채우는 완성 보드 생성용 백트래킹"""
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
    """저장된 구버전 데이터 등 정답이 없는 경우를 위한 결정론적 해법"""
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


def count_solutions(board, limit=2):
    """해의 개수를 limit까지만 세는 카운터 (유일해 검증용, 성능을 위해 조기 종료)"""
    count = 0

    def backtrack(b):
        nonlocal count
        if count >= limit:
            return
        for row in range(9):
            for col in range(9):
                if b[row][col] == 0:
                    for num in range(1, 10):
                        if is_valid(b, row, col, num):
                            b[row][col] = num
                            backtrack(b)
                            b[row][col] = 0
                            if count >= limit:
                                return
                    return
        count += 1

    backtrack([row[:] for row in board])
    return count


def generate_sudoku_puzzle(difficulty, max_attempts=200):
    """완성 보드에서 칸을 제거하되, 매 제거 시도마다 해가 유일한지 검증하여
    다중 해 퍼즐이 생성되지 않도록 보완"""
    full_board = [[0] * 9 for _ in range(9)]
    solve_board(full_board)

    clues_count = {'초급': 38, '중급': 30, '고급': 24}.get(difficulty, 30)
    remove_target = 81 - clues_count

    puzzle = [row[:] for row in full_board]
    positions = [(r, c) for r in range(9) for c in range(9)]
    # 인쇄용 PDF 생성 시에도 랜덤성을 주기 위해 positions를 한번 더 섞음
    random.shuffle(positions)

    removed = 0
    attempts = 0
    for r, c in positions:
        if removed >= remove_target or attempts >= max_attempts:
            break
        backup = puzzle[r][c]
        puzzle[r][c] = 0
        attempts += 1
        if count_solutions(puzzle, limit=2) == 1:
            removed += 1
        else:
            puzzle[r][c] = backup  # 유일해가 깨지면 되돌림

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

# [신규 추가] 인쇄용 A4 PDF를 생성하는 핵심 helper 함수
def create_daily_sudoku_pdf(puzzle, difficulty):
    """
    A4 사이즈 종이에 제목, 날짜 입력란, 그리고 가운데 정렬된 스도쿠 판을 배치한 PDF 생성
    """
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.add_page()
    
    # 폰트 설정 (기본 가변 폰트 사용, 한글 제목은 fpdf 기본 폰트 제약상 영어로 설정)
    pdf.set_font("Helvetica", 'B', 24)
    
    # 제목: Daily Sudoku Puzzle
    pdf.cell(0, 20, "Daily Sudoku Puzzle", ln=True, align='C')
    
    # 난이도 및 날짜 기입란 (약간 아래로)
    pdf.set_font("Helvetica", '', 12)
    pdf.ln(5)
    
    # 날짜 기입란: ____년 __월 __일 (한글 대신 점선으로 처리)
    # 구글 fpdf 기본 한글 폰트 미지원으로, 사용자 기입 형식으로 디자인
    pdf.cell(0, 10, f"Difficulty: {difficulty}      Date: ____ / ____ / ____", ln=True, align='C')
    
    # 스도쿠 판 그리기 (A4 가로폭 210mm, 양쪽 마진 감안하여 판 크기 설정)
    grid_size_mm = 160  # 전체 스도쿠 판 크기 (16cm x 16cm)
    cell_size_mm = grid_size_mm / 9
    
    # 판을 가운데 정렬하기 위한 시작 좌표 계산
    # (A4 가로 210 - 판크기 160) / 2 = 25mm 마진
    # (A4 세로 297 - 판크기 160 - 제목영역 약 50) / 2 -> 제목 아래 적당한 위치로 고정
    start_x = (210 - grid_size_mm) / 2
    start_y = 60 # 제목 영역 아래 6cm 지점부터 시작
    
    # 숫자 그리기 및 일반 격자
    pdf.set_font("Helvetica", 'B', 20)
    pdf.set_line_width(0.2) # 일반 격자 두께
    
    for r in range(9):
        # Y 좌표 이동
        pdf.set_xy(start_x, start_y + (r * cell_size_mm))
        for c in range(9):
            val = puzzle[r][c]
            txt = str(val) if val != 0 else ""
            
            # 셀 그리기 (테두리 포함)
            pdf.cell(cell_size_mm, cell_size_mm, txt, border=1, ln=0, align='C')
            
    # 3x3 박스 굵은 테두리 덧그리기 (X, Y축 반복문 활용)
    pdf.set_line_width(0.8) # 3x3 굵은 테두리 두께
    pdf.set_draw_color(34, 34, 34) # 약간 짙은 회색/검정
    
    # 굵은 가로선
    for i in range(4): # 0, 3, 6, 9 인덱스 위치
        pdf.line(start_x, start_y + (i * 3 * cell_size_mm), 
                 start_x + grid_size_mm, start_y + (i * 3 * cell_size_mm))
                 
    # 굵은 세로선
    for i in range(4): # 0, 3, 6, 9 인덱스 위치
        pdf.line(start_x + (i * 3 * cell_size_mm), start_y, 
                 start_x + (i * 3 * cell_size_mm), start_y + grid_size_mm)
                 
    # PDF 데이터를 메모리 버퍼로 반환
    # fpdf2 최신 버전에서는 output(dest='S') 대신 bytes 반환 방식을 권장
    pdf_bytes = pdf.output()
    return pdf_bytes


# ------------------------------------------------------------------------------
# 4. 메인 UI 구조
# ------------------------------------------------------------------------------
st.title("🧩 스도쿠 AI 스마트 도우미")

tab1, tab2 = st.tabs(["📸 이미지 업로드 & 도움받기", "🎲 문제 만들기 & 보관함"])

# ==============================================================================
# TAB 1: 이미지 업로드 & 도움받기
# ==============================================================================
with tab1:
    st.subheader("1. 스도쿠 이미지 가져오기")
    img_file = st.file_uploader("스도쿠 이미지를 촬영하거나 업로드하세요", type=["jpg", "jpeg", "png"])

    if img_file is not None:
        file_bytes = img_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()

        # 파일명이 같아도 실제 내용이 다르면(재촬영 등) 새 업로드로 인식
        if st.session_state.get("last_uploaded_file_hash") != file_hash:
            st.session_state["last_uploaded_file_hash"] = file_hash
            st.session_state["rotate_angle"] = 0
            st.session_state.pop("ai_analysis_result", None)
            st.session_state.pop("last_cropped_hash", None)

        try:
            raw_image = Image.open(img_file)
            raw_image.verify()
            raw_image = Image.open(img_file)  # verify() 이후 재오픈 필요
        except Exception:
            st.error("이미지 파일을 열 수 없습니다. 다른 파일을 업로드해 주세요.")
            st.stop()

        working_img = compress_for_display(raw_image, max_dim=350)

        if "rotate_angle" not in st.session_state:
            st.session_state["rotate_angle"] = 0

        st.subheader("2. 사진 방향 및 영역 설정")
        col_rot1, col_rot2 = st.columns(2)
        with col_rot1:
            if st.button("🔄 90° 회전"):
                st.session_state["rotate_angle"] = (st.session_state["rotate_angle"] - 90) % 360
                st.session_state.pop("ai_analysis_result", None)
        with col_rot2:
            if st.button("↩️ 방향 초기화"):
                st.session_state["rotate_angle"] = 0
                st.session_state.pop("ai_analysis_result", None)

        if st.session_state["rotate_angle"] != 0:
            working_img = working_img.rotate(st.session_state["rotate_angle"], expand=True)

        use_cropper = st.checkbox("✂️ 빨간 박스로 9x9 잘라내기 사용", value=True)
        if st.session_state.get("last_use_cropper") != use_cropper:
            st.session_state["last_use_cropper"] = use_cropper
            st.session_state.pop("ai_analysis_result", None)

        target_img = working_img
        if use_cropper:
            st.write("📱 모서리를 조절해 9x9 영역에 맞추세요.")
            target_img = st_cropper(
                working_img,
                realtime_update=True,
                box_color='#FF0000',
                aspect_ratio=(1, 1),
                key="cropper_widget",
            )

            # PIL 이미지는 identity 비교라 항상 "다름"으로 판정되는 문제가 있어
            # 픽셀 해시로 실제 변경 여부를 판단
            current_hash = image_hash(target_img)
            if st.session_state.get("last_cropped_hash") != current_hash:
                st.session_state["last_cropped_hash"] = current_hash
                st.session_state.pop("ai_analysis_result", None)

        if target_img:
            st.image(target_img, caption="최종 분석 영역", use_container_width=True)

            if st.button("💡 도움받기 (단 하나의 힌트 & 검증)", type="primary"):
                with st.spinner("Gemini AI가 정밀 분석 중입니다..."):
                    system_prompt = """
                    당신은 엄격하고 명확한 스도쿠 검증 튜터입니다.
                    업로드된 이미지에서 9x9 스도쿠 판(인쇄체 및 손글씨)을 분석하세요.

                    [주의 규칙]
                    1. errors: 손글씨 중 스도쿠 규칙에 위배되는(틀린) 숫자의 위치(row, col)를 기록하세요. 없으면 빈 배열을 반환하세요.
                    2. single_hint: 현재 확실히 바로 채울 수 있는 '단 한 칸'의 위치, 정답 숫자, 이유를 제시하세요. 확신이 없으면 null로 두세요.
                    3. row, col은 1~9 범위의 정수입니다.
                    """

                    # 화면 표시용(저해상도)과 별개로, API 전송에는 고해상도 버전 사용
                    api_img = prepare_for_api(target_img, max_dim=768)

                    try:
                        response = client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=[api_img, "이 스도쿠 판의 틀린 손글씨 위치와 바로 해결 가능한 단 하나의 힌트를 JSON으로 출력하세요."],
                            config=types.GenerateContentConfig(
                                system_instruction=system_prompt,
                                temperature=0.1,
                                response_mime_type="application/json",
                                response_schema=SudokuAnalysis,
                            ),
                        )

                        if response and response.text:
                            try:
                                parsed = SudokuAnalysis.model_validate_json(response.text)
                                st.session_state["ai_analysis_result"] = parsed.model_dump()
                                st.session_state["cropped_img_for_display"] = target_img
                            except Exception as parse_err:
                                st.error(f"AI 응답 형식이 올바르지 않습니다: {parse_err}")
                        else:
                            st.error("분석 응답을 불러오지 못했습니다. 다시 한번 시도해 주세요.")

                    except json.JSONDecodeError:
                        st.error("AI가 유효한 JSON을 반환하지 않았습니다. 다시 시도해 주세요.")
                    except Exception as e:
                        st.error(f"API 호출 중 오류가 발생했습니다: {e}")

            if "ai_analysis_result" in st.session_state and "cropped_img_for_display" in st.session_state:
                result = st.session_state["ai_analysis_result"]
                saved_img = st.session_state["cropped_img_for_display"]

                errors = result.get("errors", [])
                hint = result.get("single_hint")

                st.markdown("---")

                if errors:
                    st.error(f"⚠️ **검증 결과:** 손글씨 중 틀린 부분이 {len(errors)}곳 발견되었습니다!")
                    annotated_image = draw_errors_on_image(saved_img, errors)
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

# ==============================================================================
# TAB 2: 스도쿠 문제 만들기 & 보관함
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
        with st.spinner("유일해를 가지는 문제를 생성하는 중입니다..."):
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
            format_func=lambda i: f"#{saved_puzzles[i]['id']} [{saved_puzzles[i]['difficulty']}] ({saved_puzzles[i].get('created_at', '')})",
        )

        p_data = saved_puzzles[selected_idx]
        pz_saved = p_data["puzzle"]
        sol_saved = p_data.get("solution") or solve_sudoku_exact(pz_saved)

        # 보관함 하단 버튼 배치 (기존 정답 토글 유지)
        col_saved1, col_saved2 = st.columns([2, 1])
        
        with col_saved1:
            show_saved_sol = st.toggle("🔍 저장된 문제 정답 보기", key="saved_sol_toggle")
            sol_param = sol_saved if show_saved_sol else None
            st.markdown(render_sudoku_board_html(pz_saved, solution=sol_param), unsafe_allow_html=True)
            
        # [신규 추가] PDF 다운로드 버튼 배치
        with col_saved2:
            st.write("") # 간격 맞춤용
            st.write("") # 간격 맞춤용
            with st.spinner("PDF를 준비하는 중..."):
                # 현재 선택된 문제 데이터를 기반으로 PDF 바이너리 데이터 생성
                pdf_bytes = create_daily_sudoku_pdf(pz_saved, p_data['difficulty'])
                
                # 파일명 설정: daily_sudoku_초급_20240315.pdf 형식
                date_str = datetime.datetime.now().strftime("%Y%m%d")
                file_name = f"daily_sudoku_{p_data['difficulty']}_{date_str}.pdf"
                
                # Streamlit 다운로드 버튼 UI 배치
                st.download_button(
                    label="📄 인쇄용 PDF 다운로드",
                    data=pdf_bytes,
                    file_name=file_name,
                    mime="application/pdf",
                    key=f"pdf_down_{p_data['id']}_{show_saved_sol}" # 고유 키 설정
                )
    else:
        st.info("선택한 난이도에 저장된 스도쿠 문제가 없습니다. 위에서 새 문제를 만들어 보세요!")
