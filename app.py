import os
import json
import random
import hashlib
import datetime
import streamlit as st
from PIL import Image, ImageDraw, ImageOps
from streamlit_cropper import st_cropper
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

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
# 3. 핵심 알고리즘 (생성/검증/시각화)
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
