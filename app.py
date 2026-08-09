import os
import io
import json
import random
import hashlib
import datetime as dt
from zoneinfo import ZoneInfo

import gspread
from google.oauth2.service_account import Credentials

import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps
from streamlit_cropperjs import st_cropperjs
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas
from reportlab.pdfbase.pdfmetrics import stringWidth

# =============================================================================
# 구글 시트 연동
# =============================================================================
SHEET_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# 시트 컬럼 순서: id, difficulty, puzzle, solution, created_at, source, solved
SOLVED_COLUMN_INDEX = 7


@st.cache_resource
def get_sheet_client():
    credentials = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=SHEET_SCOPES,
    )
    return gspread.authorize(credentials)


@st.cache_resource
def get_worksheet():
    client = get_sheet_client()
    spreadsheet = client.open_by_url(st.secrets["sheets"]["spreadsheet_url"])
    return spreadsheet.worksheet(st.secrets["sheets"]["worksheet_name"])


# =============================================================================
# 설정
# =============================================================================
st.set_page_config(page_title="영용's Sudoku", page_icon="🏄", layout="centered")


def check_password():
    def password_entered():
        if st.session_state["password"] == st.secrets["auth"]["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.text_input("비밀번호를 입력하세요", type="password", on_change=password_entered, key="password")

    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("비밀번호가 틀렸습니다.")

    return False


if not check_password():
    st.stop()

st.markdown("""
<style>
.stApp {max-width:100%; padding-left:.5rem; padding-right:.5rem;}
.app-title {text-align:center; margin:.4rem 0 1.4rem;}
.sudoku-wrap {display:flex; justify-content:center; overflow-x:auto; margin:15px 0;}
.sudoku {border-collapse:collapse; border:3px solid #222; background:#fff;}
.sudoku td {width:36px; height:36px; border:1px solid #ccc; text-align:center; vertical-align:middle; font-size:18px; font-weight:700; color:#111;}
.sudoku tr:nth-child(3n) td {border-bottom:2px solid #222;}
.sudoku td:nth-child(3n) {border-right:2px solid #222;}
.sudoku tr:first-child td {border-top:2px solid #222;}
.sudoku td:first-child {border-left:2px solid #222;}
.sudoku td.error {background:#fecaca; color:#991b1b;}
.sudoku td.hint {background:#fef3c7;}
.sudoku td.answer {background:#eff6ff; color:#1d4ed8;}
</style>
""", unsafe_allow_html=True)
st.markdown("<h1 class='app-title'>🏄영용's Sudoku</h1>", unsafe_allow_html=True)

MAX_IMAGE_DIM = 768
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")


# =============================================================================
# Gemini AI 데이터 형식
# =============================================================================
class SudokuError(BaseModel):
    row: int = Field(ge=1, le=9)
    col: int = Field(ge=1, le=9)
    reason: str

class SudokuHint(BaseModel):
    row: int = Field(ge=1, le=9)
    col: int = Field(ge=1, le=9)
    number: int = Field(ge=1, le=9)
    reason: str

class SudokuAnalysis(BaseModel):
    grid: list[list[int]]
    errors: list[SudokuError] = Field(default_factory=list)
    single_hint: SudokuHint | None = None


@st.cache_resource
def get_client(api_key):
    return genai.Client(api_key=api_key)

def api_key_value():
    return st.secrets.get("GEMINI_API_KEY", None) or os.getenv("GEMINI_API_KEY")

def validate_grid(grid):
    if len(grid) != 9 or any(len(row) != 9 for row in grid):
        raise ValueError("AI가 9x9 형식이 아닌 데이터를 반환했습니다.")
    if any(not isinstance(n, int) or n < 0 or n > 9 for row in grid for n in row):
        raise ValueError("스도쿠 숫자는 0~9의 정수여야 합니다.")
    return grid

def ai_read_grid(client, image, model_name, instruction, prompt_text):
    response = client.models.generate_content(
        model=model_name,
        contents=[image, prompt_text],
        config=types.GenerateContentConfig(
            system_instruction=instruction,
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=SudokuAnalysis,
        ),
    )
    if not response or not response.text:
        raise RuntimeError("Gemini가 비어 있는 응답을 반환했습니다.")
    result = SudokuAnalysis.model_validate_json(response.text)
    result.grid = validate_grid(result.grid)
    return result

def ai_read_sudoku(client, image, model_name):
    instruction = """
당신은 스도쿠 사진 판독 전문 AI입니다. 사진 속 9x9 스도쿠를 읽어라.
인쇄 숫자와 손글씨 숫자를 모두 읽어 grid에 기록하라.
- grid는 9개의 행과 각 행의 9개 숫자로 구성된다.
- 빈칸 또는 확신할 수 없는 숫자는 0으로 기록하고 추측하지 마시오.
- errors와 single_hint도 스키마에 맞게 반환하라.
- 격자 밖의 텍스트, 손글씨 중 비교적 작은 것, 메모는 무시하라.
"""
    return ai_read_grid(
        client, image, model_name, instruction,
        "사진의 스도쿠를 읽어 9x9 JSON 데이터로 반환하세요."
    )

def ai_read_puzzle_only(client, image, model_name):
    instruction = """
당신은 스도쿠 문제집 판독 전문 AI입니다. 사진 속 9x9 스도쿠 '문제'(아직 풀지 않은 상태)를 읽으세요.
- 인쇄된 숫자만 grid에 기록하세요.
- 빈칸은 0으로 기록하세요.
- 확신할 수 없는 숫자는 추측하지 말고 0으로 두세요.
- errors, single_hint는 사용하지 않으니 신경쓰지 않아도 됩니다.
"""
    return ai_read_grid(
        client, image, model_name, instruction,
        "이 사진 속 스도쿠 문제를 9x9 JSON grid로만 반환하세요."
    )


# =============================================================================
# 이미지 함수
# =============================================================================
def normalize(image):
    return ImageOps.exif_transpose(image).convert("RGB")

def resize_image(image, max_dim=MAX_IMAGE_DIM):
    image = normalize(image)
    width, height = image.size
    if max(width, height) <= max_dim:
        return image
    ratio = max_dim / max(width, height)
    return image.resize((int(width * ratio), int(height * ratio)), Image.Resampling.LANCZOS)

def image_hash(image):
    image = normalize(image)
    return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()

def draw_photo_x(image, cells):
    result = normalize(image).copy()
    draw = ImageDraw.Draw(result)
    width, height = result.size
    cell_w, cell_h = width / 9, height / 9
    line_w = max(3, int(width / 80))
    for row, col in cells:
        x1, y1 = col * cell_w + cell_w*.15, row * cell_h + cell_h*.15
        x2, y2 = (col+1)*cell_w - cell_w*.15, (row+1)*cell_h - cell_h*.15
        draw.line((x1,y1,x2,y2), fill="red", width=line_w)
        draw.line((x1,y2,x2,y1), fill="red", width=line_w)
    return result


# =============================================================================
# 스도쿠 알고리즘
# =============================================================================
def is_valid(board, row, col, number):
    for index in range(9):
        if board[row][index] == number or board[index][col] == number:
            return False
        if board[3*(row//3)+index//3][3*(col//3)+index%3] == number:
            return False
    return True

def candidates(board, row, col):
    if board[row][col] != 0:
        return []
    return [number for number in range(1, 10) if is_valid(board, row, col, number)]

def find_empty(board):
    best = None
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                values = candidates(board, row, col)
                if not values:
                    return row, col, []
                if best is None or len(values) < len(best[2]):
                    best = (row, col, values)
    return best

def solve(board):
    empty = find_empty(board)
    if empty is None:
        return True
    row, col, values = empty
    for number in values:
        board[row][col] = number
        if solve(board):
            return True
        board[row][col] = 0
    return False

def count_solutions(board, limit=2):
    empty = find_empty(board)
    if empty is None:
        return 1
    row, col, values = empty
    total = 0
    for number in values:
        board[row][col] = number
        total += count_solutions(board, limit)
        board[row][col] = 0
        if total >= limit:
            return total
    return total

def make_random_solution():
    """유효한 완성 스도쿠 판을 빠르게 만들고 무작위 변환한다."""
    base = [
        [(row * 3 + row // 3 + col) % 9 + 1 for col in range(9)]
        for row in range(9)
    ]

    digits = list(range(1, 10))
    random.shuffle(digits)
    board = [[digits[value - 1] for value in row] for row in base]

    bands = [0, 1, 2]
    random.shuffle(bands)
    rows = []
    for band in bands:
        inner_rows = [0, 1, 2]
        random.shuffle(inner_rows)
        rows.extend([band * 3 + item for item in inner_rows])
    board = [board[row][:] for row in rows]

    stacks = [0, 1, 2]
    random.shuffle(stacks)
    cols = []
    for stack in stacks:
        inner_cols = [0, 1, 2]
        random.shuffle(inner_cols)
        cols.extend([stack * 3 + item for item in inner_cols])
    board = [[row[col] for col in cols] for row in board]

    if random.choice([True, False]):
        board = [[board[col][row] for col in range(9)] for row in range(9)]

    return board


def clue_count(board):
    """현재 문제에 남아 있는 숫자 개수를 반환한다."""
    return sum(value != 0 for row in board for value in row)


def make_rotational_groups():
    """
    180도 회전 대칭 위치를 묶는다.
    예: (0, 0)과 (8, 8)을 한 쌍으로 제거한다.
    중앙 (4, 4)은 단독 그룹이다.
    """
    groups = []
    visited = set()
    for row in range(9):
        for col in range(9):
            if (row, col) in visited:
                continue
            opposite = (8 - row, 8 - col)
            group = [(row, col)]
            if opposite != (row, col):
                group.append(opposite)
            visited.update(group)
            groups.append(group)
    return groups


def try_remove_group(puzzle, group):
    """한 그룹을 비운 뒤, 유일한 해가 유지될 때만 제거를 확정한다."""
    backup = [(row, col, puzzle[row][col]) for row, col in group]
    for row, col, _ in backup:
        puzzle[row][col] = 0

    if count_solutions([row[:] for row in puzzle], limit=2) == 1:
        return True

    for row, col, value in backup:
        puzzle[row][col] = value
    return False


def generate_puzzle(difficulty, max_board_attempts=8):
    needed = {"초급": 38, "중급": 30, "고급": 24}[difficulty]

    best_puzzle, best_answer, best_left = None, None, 82

    for _ in range(max_board_attempts):
        answer = make_random_solution()
        puzzle = [row[:] for row in answer]

        groups = make_rotational_groups()
        random.shuffle(groups)

        left = 81
        for group in groups:
            if left - len(group) < needed:
                continue
            if try_remove_group(puzzle, group):
                left -= len(group)
            if left <= needed:
                break

        if left > needed:
            positions = [(r, c) for r in range(9) for c in range(9) if puzzle[r][c] != 0]
            random.shuffle(positions)
            for r, c in positions:
                if left <= needed:
                    break
                old = puzzle[r][c]
                puzzle[r][c] = 0
                if count_solutions([row[:] for row in puzzle], limit=2) == 1:
                    left -= 1
                else:
                    puzzle[r][c] = old

        if left == needed:
            return puzzle, answer

        if left < best_left:
            best_puzzle, best_answer, best_left = puzzle, answer, left

    return best_puzzle, best_answer


def find_rule_errors(board):
    errors = set()
    def inspect(cells):
        values = {}
        for row, col in cells:
            if board[row][col] != 0:
                values.setdefault(board[row][col], []).append((row, col))
        for duplicate_cells in values.values():
            if len(duplicate_cells) > 1:
                errors.update(duplicate_cells)
    for row in range(9):
        inspect([(row, col) for col in range(9)])
    for col in range(9):
        inspect([(row, col) for row in range(9)])
    for box_row in range(0, 9, 3):
        for box_col in range(0, 9, 3):
            inspect([(row, col) for row in range(box_row,box_row+3) for col in range(box_col,box_col+3)])
    return errors

def find_hint_cells(board):
    hints = set()
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0 and len(candidates(board,row,col)) == 1:
                hints.add((row,col))
    groups = []
    groups += [[(row,col) for col in range(9)] for row in range(9)]
    groups += [[(row,col) for row in range(9)] for col in range(9)]
    groups += [[(row,col) for row in range(br,br+3) for col in range(bc,bc+3)] for br in range(0,9,3) for bc in range(0,9,3)]
    for group in groups:
        possible = {number:[] for number in range(1,10)}
        for row,col in group:
            if board[row][col] == 0:
                for number in candidates(board,row,col):
                    possible[number].append((row,col))
        for cells in possible.values():
            if len(cells) == 1:
                hints.add(cells[0])
    return hints


# =============================================================================
# 직접 입력 / 사진 인식 공통 검증
# =============================================================================
def parse_manual_board(cells):
    """9x9 입력 위젯 값들을 보드 리스트로 변환한다."""
    return [[cells[row][col] for col in range(9)] for row in range(9)]

def parse_row_strings(row_texts):
    """
    9줄짜리 문자열 리스트(각 줄 9자리 숫자)를 9x9 보드로 변환한다.
    실패 시 (None, 오류_메시지)를 반환한다.
    """
    board = []
    for index, text in enumerate(row_texts):
        cleaned = text.strip().replace(" ", "")
        if len(cleaned) != 9 or not cleaned.isdigit():
            return None, f"{index + 1}행: 9자리 숫자(0~9)로 입력해야 합니다. 지금 입력: '{text}'"
        board.append([int(ch) for ch in cleaned])
    return board, None
    
def parse_row_strings(row_texts):
    """
    9줄짜리 문자열 리스트(각 줄 9자리 숫자)를 9x9 보드로 변환한다.
    실패 시 (None, 오류_메시지)를 반환한다.
    """
    board = []
    for index, text in enumerate(row_texts):
        cleaned = text.strip().replace(" ", "")
        if len(cleaned) != 9 or not cleaned.isdigit():
            return None, f"{index + 1}행: 9자리 숫자(0~9)로 입력해야 합니다. 지금 입력: '{text}'"
        board.append([int(ch) for ch in cleaned])
    return board, None


def validate_manual_board(board):
    """
    수동/사진 입력 판을 검증한다.
    반환값: (오류_메시지 또는 None, 완성된_정답판 또는 None)
    """
    errors = find_rule_errors(board)
    if errors:
        return "행/열/3×3 박스 중 중복된 숫자가 있습니다. 입력을 다시 확인하세요.", None

    filled = clue_count(board)
    if filled < 17:
        return "단서가 너무 적습니다. 최소 17개 이상의 숫자가 필요합니다.", None

    solved_board = [row[:] for row in board]
    if not solve(solved_board):
        return "이 판은 해가 존재하지 않습니다. 숫자를 다시 확인하세요.", None

    solutions = count_solutions([row[:] for row in board], limit=2)
    if solutions != 1:
        return "이 판은 정답이 하나로 정해지지 않습니다(다중 해). 단서를 더 확인해 주세요.", None

    return None, solved_board


# =============================================================================
# 보관함 (구글 시트)
# =============================================================================
def load_puzzles(difficulty=None):
    try:
        worksheet = get_worksheet()
        rows = worksheet.get_all_records()
    except Exception as error:
        st.warning(f"보관함을 불러오지 못했습니다: {error}")
        return []

    items = []
    for row in rows:
        try:
            items.append({
                "id": int(row["id"]),
                "difficulty": row["difficulty"],
                "puzzle": json.loads(row["puzzle"]),
                "solution": json.loads(row["solution"]),
                "created_at": row["created_at"],
                "source": row.get("source") or "생성",
                "solved": row.get("solved") or "",
            })
        except (KeyError, ValueError, json.JSONDecodeError):
            continue

    if difficulty:
        items = [item for item in items if item["difficulty"] == difficulty]

    return items


def save_puzzle(difficulty, puzzle, answer, source="생성"):
    worksheet = get_worksheet()
    existing = load_puzzles()
    new_id = max([item["id"] for item in existing], default=0) + 1

    created_at = dt.datetime.now(
        ZoneInfo("Asia/Seoul")
    ).strftime("%Y-%m-%d %H:%M:%S KST")

    worksheet.append_row([
        new_id,
        difficulty,
        json.dumps(puzzle),
        json.dumps(answer),
        created_at,
        source,
        "",  # solved - 아직 안 풀림
    ])
    return new_id


def mark_puzzle_solved(item_id):
    """문제를 다 맞혔을 때 보관함에 풀이 완료 시각을 기록한다."""
    try:
        worksheet = get_worksheet()
        cell = worksheet.find(str(item_id), in_column=1)
        if cell is None:
            return False
        solved_at = dt.datetime.now(
            ZoneInfo("Asia/Seoul")
        ).strftime("%Y-%m-%d %H:%M:%S KST")
        worksheet.update_cell(cell.row, SOLVED_COLUMN_INDEX, solved_at)
        return True
    except Exception:
        return False


# =============================================================================
# 화면 표 / PDF / PNG
# =============================================================================
def render_board(board, answer=None, errors=None, hints=None):
    errors, hints = errors or set(), hints or set()
    html = "<div class='sudoku-wrap'><table class='sudoku'>"
    for row in range(9):
        html += "<tr>"
        for col in range(9):
            value = board[row][col]
            css = "error" if (row,col) in errors else "hint" if (row,col) in hints else "answer" if value == 0 and answer else ""
            shown = value or (answer[row][col] if answer else "")
            html += f"<td class='{css}'>{shown}</td>"
        html += "</tr>"
    return html + "</table></div>"

def load_font(size, bold=False):
    paths = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"]
    for path in paths:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

def make_png(board, title="Daily Sudoku Puzzle"):
    cell, margin, title_h = 72, 42, 72
    size = cell * 9
    image = Image.new("RGB", (size+margin*2, size+margin*2+title_h), "white")
    draw = ImageDraw.Draw(image)
    draw.text((margin,18), title, fill="#111827", font=load_font(28, True))
    x0, y0 = margin, margin+title_h
    for index in range(10):
        width = 5 if index % 3 == 0 else 1
        draw.line((x0+index*cell,y0,x0+index*cell,y0+size),fill="#111",width=width)
        draw.line((x0,y0+index*cell,x0+size,y0+index*cell),fill="#111",width=width)
    number_font = load_font(38, True)
    for row in range(9):
        for col in range(9):
            if board[row][col]:
                text = str(board[row][col]); box = draw.textbbox((0,0),text,font=number_font)
                draw.text((x0+col*cell+(cell-(box[2]-box[0]))/2,y0+row*cell+(cell-(box[3]-box[1]))/2-4),text,fill="#111",font=number_font)
    data = io.BytesIO(); image.save(data,format="PNG",optimize=True)
    return data.getvalue()

def make_pdf(board, date_value, difficulty):
    data = io.BytesIO()
    pdf = canvas.Canvas(data, pagesize=A4)

    width, height = A4

    size = 99 * mm
    cell = size / 9

    left = (width - size) / 2
    bottom = (height - size) / 2

    pdf.setFillColor(HexColor("#111827"))
    pdf.setFont("Helvetica-Bold", 26)
    pdf.drawCentredString(width / 2, height - 30 * mm, "Daily Sudoku Puzzle")

    pdf.setFillColor(HexColor("#4B5563"))
    pdf.setFont("Helvetica", 12)
    pdf.drawCentredString(width / 2, height - 39 * mm, date_value.strftime("%Y.%m.%d"))

    count = {"초급": 1, "중급": 2, "고급": 3}.get(difficulty, 1)
    square = 5 * mm
    gap = 2 * mm
    start = (width - (count * square + (count - 1) * gap)) / 2
    y = height - 49 * mm

    pdf.setFillColor(HexColor("#2563EB"))
    for index in range(count):
        pdf.roundRect(start + index * (square + gap), y, square, square, 1.2 * mm, stroke=0, fill=1)

    pdf.setStrokeColor(HexColor("#111111"))
    for index in range(10):
        pdf.setLineWidth(1.5 if index % 3 == 0 else 0.35)
        position = index * cell
        pdf.line(left + position, bottom, left + position, bottom + size)
        pdf.line(left, bottom + position, left + size, bottom + position)

    pdf.setFillColor(HexColor("#111111"))
    pdf.setFont("Helvetica-Bold", 15)
    for row in range(9):
        for col in range(9):
            if board[row][col]:
                text = str(board[row][col])
                x = left + col * cell + (cell - stringWidth(text, "Helvetica-Bold", 15)) / 2
                y = bottom + (8 - row) * cell + cell * 0.30
                pdf.drawString(x, y, text)

    pdf.setFillColor(HexColor("#6B7280"))
    pdf.setFont("Helvetica", 9)
    pdf.drawCentredString(width / 2, 24 * mm, "Solve one square at a time. Enjoy your puzzle!")

    pdf.save()
    return data.getvalue()

def download_buttons(board,difficulty,prefix):
    st.caption("PDF와 PNG는 Google Drive가 아니라 현재 사용 중인 기기에 저장됩니다.")
    date_value=st.date_input("인쇄 날짜",value=dt.date.today(),key=prefix+"_date"); stamp=date_value.strftime("%Y%m%d")
    left,right=st.columns(2)
    left.download_button("🖨️ A4 PDF 저장",make_pdf(board,date_value,difficulty),f"daily_sudoku_{stamp}.pdf","application/pdf",key=prefix+"_pdf",use_container_width=True)
    right.download_button("🖼️ PNG 저장",make_png(board,f"Daily Sudoku Puzzle · {difficulty}"),f"daily_sudoku_{stamp}.png","image/png",key=prefix+"_png",use_container_width=True)


def archive_label(item):
    solved_mark = "✅ " if item.get("solved") else "⬜ "
    return f"{solved_mark}#{item.get('id','?')} [{item.get('difficulty','')}] ({item.get('source','생성')}) {item.get('created_at','')}"


# =============================================================================
# 탭 구성
# =============================================================================
tab_read, tab_create, tab_manual, tab_solve = st.tabs([
    "📸 사진 읽기 & 확인",
    "🎲 문제 만들기 & 보관함",
    "✍️ 문제 입력",
    "📝 문제 풀이",
])

# =============================================================================
# 탭 1: 모바일 확대/이동/자르기 및 AI 판독 (손글씨 검증)
# =============================================================================
with tab_read:
    st.subheader("1. 스도쿠 사진 가져오기")
    upload=st.file_uploader("스도쿠 사진을 촬영하거나 업로드하세요",type=["jpg","jpeg","png"])
    if upload is not None:
        file_hash=hashlib.sha256(upload.getvalue()).hexdigest()
        if st.session_state.get("upload_hash") != file_hash:
            st.session_state["upload_hash"]=file_hash; st.session_state["rotate_angle"]=0
            for key in ["crop_hash","analysis","analysis_image","celebrated"]: st.session_state.pop(key,None)
        try:
            raw=normalize(Image.open(upload))
        except Exception:
            st.error("이미지를 열 수 없습니다."); st.stop()
        st.session_state.setdefault("rotate_angle",0)
        st.subheader("2. 사진 확대·이동 및 9×9 영역 자르기")
        col1,col2=st.columns(2)
        with col1:
            if st.button("🔄 90° 회전"):
                st.session_state["rotate_angle"]=(st.session_state["rotate_angle"]-90)%360
                st.session_state.pop("analysis",None)
        with col2:
            if st.button("↩️ 방향 초기화"):
                st.session_state["rotate_angle"]=0
                st.session_state.pop("analysis",None)
        work=resize_image(raw)
        if st.session_state["rotate_angle"]:
            work=work.rotate(st.session_state["rotate_angle"],expand=True)
        use_crop=st.checkbox("✂️ 9×9 영역 자르기",value=True,key="use_mobile_crop")
        if use_crop:
            st.info("📱 사진을 손가락으로 확대·축소하거나 이동한 뒤, 9×9 격자만 보이도록 자르세요. 완료 버튼을 누르면 영역이 확정됩니다.")
            source=io.BytesIO(); work.save(source,format="PNG")
            cropped_bytes=st_cropperjs(pic=source.getvalue(),btn_text="✅ 이 영역으로 확정",key="mobile_cropper")
            target=normalize(Image.open(io.BytesIO(cropped_bytes))) if cropped_bytes else None
        else:
            target=work
        if target is not None:
            crop_hash=image_hash(target)
            if st.session_state.get("crop_hash") != crop_hash:
                st.session_state["crop_hash"]=crop_hash
                for key in ["analysis","analysis_image","celebrated"]: st.session_state.pop(key,None)
            st.image(target,caption="AI가 읽을 최종 9×9 영역",use_container_width=True)
            api_key=api_key_value(); model=st.text_input("Gemini 모델",value=DEFAULT_MODEL,key="model_name")
            if not api_key:
                st.info("Streamlit Secrets에 GEMINI_API_KEY를 설정해 주세요.")
            elif st.button("🔎 손글씨 읽기 및 정답 확인",type="primary"):
                try:
                    with st.spinner("손글씨를 읽고 스도쿠 규칙을 확인하고 있습니다..."):
                        result=ai_read_sudoku(get_client(api_key),target,model)
                    st.session_state["analysis"]=result.model_dump(); st.session_state["analysis_image"]=target.copy()
                except Exception as error:
                    st.error("AI 분석에 실패했습니다. API 키, 모델명, 사용 권한을 확인해 주세요."); st.exception(error)
        if "analysis" in st.session_state:
            grid=validate_grid(SudokuAnalysis.model_validate(st.session_state["analysis"]).grid)
            errors=find_rule_errors(grid); complete=all(number != 0 for row in grid for number in row)
            st.markdown("---"); st.subheader("🔎 AI가 읽은 9×9 스도쿠 판")
            if errors:
                st.markdown(render_board(grid,errors=errors),unsafe_allow_html=True)
                st.error("규칙에 맞지 않는 숫자를 빨간색으로 표시했습니다.")
                if "analysis_image" in st.session_state:
                    st.image(draw_photo_x(st.session_state["analysis_image"],errors),caption="사진에서 오류가 있는 위치",use_container_width=True)
            elif complete:
                st.markdown(render_board(grid),unsafe_allow_html=True)
                st.success("🎉 정답입니다! 모든 행, 열, 3×3 박스가 규칙을 만족합니다.")
                finish_hash=hashlib.sha256(json.dumps(grid).encode()).hexdigest()
                if st.session_state.get("celebrated") != finish_hash:
                    st.session_state["celebrated"]=finish_hash; st.balloons()
            else:
                hints=find_hint_cells(grid)
                st.markdown(render_board(grid,hints=hints),unsafe_allow_html=True)
                st.info("💡 노란색으로 표시된 빈칸은 현재 상태에서 바로 해결할 수 있는 칸입니다." if hints else "현재는 바로 확정할 수 있는 빈칸을 찾지 못했습니다.")
            st.download_button("🖼️ 인식된 9×9 판 PNG 저장",make_png(grid,"AI Read Sudoku Grid"),"recognized_sudoku_grid.png","image/png",key="read_png",use_container_width=True)

# =============================================================================
# 탭 2: 문제 생성 및 보관함
# =============================================================================
with tab_create:
    st.subheader("🎲 난이도별 스도쿠 문제 생성")
    first, second = st.columns([2, 1])
    difficulty = first.selectbox("난이도", ["초급", "중급", "고급"])
    second.write("")
    second.write("")

    if second.button("문제 생성", type="primary", use_container_width=True):
        with st.spinner("유일한 정답을 가진 문제를 만들고 있습니다..."):
            puzzle, answer = generate_puzzle(difficulty)

        st.session_state.update({"puzzle": puzzle, "answer": answer, "difficulty": difficulty})

        try:
            save_puzzle(difficulty, puzzle, answer, source="생성")
            st.success("문제를 만들고 보관함에 저장했습니다.")
        except Exception as error:
            st.warning(f"문제는 생성됐지만 보관함 저장에는 실패했습니다: {error}")

    if "puzzle" in st.session_state:
        st.markdown(f"### 📋 생성된 문제 · {st.session_state['difficulty']}")
        show = st.toggle("🔍 정답 보기", key="current_solution")
        st.markdown(
            render_board(st.session_state["puzzle"], st.session_state["answer"] if show else None),
            unsafe_allow_html=True
        )
        download_buttons(st.session_state["puzzle"], st.session_state["difficulty"], "new")

    st.markdown("---")
    st.subheader("📁 저장된 문제 보관함")
    selected = st.radio("조회 난이도", ["전체", "초급", "중급", "고급"], horizontal=True)
    items = load_puzzles(None if selected == "전체" else selected)

    if not items:
        st.info("저장된 문제가 없습니다.")
    else:
        index = st.selectbox(
            "불러올 문제",
            range(len(items)),
            format_func=lambda i: archive_label(items[i])
        )
        item = items[index]
        saved = item["puzzle"]
        solution = item.get("solution")

        if not solution:
            solution = [row[:] for row in saved]
            solve(solution)

        show = st.toggle("🔍 저장된 문제 정답 보기", key="saved_solution")
        st.markdown(render_board(saved, solution if show else None), unsafe_allow_html=True)
        download_buttons(saved, item.get("difficulty", "초급"), f"saved_{item.get('id', index)}")

# =============================================================================
# 탭 3: 문제 입력 (직접 입력 / 사진으로 인식)
# =============================================================================
with tab_manual:
    st.subheader("✍️ 출판된 스도쿠 문항 입력")
    st.caption("책이나 신문에 실린 스도쿠를 직접 타이핑하거나, 사진을 찍어 AI로 읽어낸 뒤 보관함에 저장합니다.")

    input_method = st.radio(
        "입력 방식",
        ["✍️ 숫자 직접 입력", "📸 사진으로 인식"],
        horizontal=True,
        key="manual_input_method"
    )

    manual_difficulty = st.selectbox(
        "난이도 표시",
        ["초급", "중급", "고급"],
        key="manual_difficulty"
    )

    # -------------------------------------------------------------------
    # 방식 1: 숫자 직접 입력 (한 줄에 9자리씩, 9줄)
    # -------------------------------------------------------------------
        if input_method == "✍️ 숫자 직접 입력":
        st.write("한 줄에 9자리씩, 빈칸은 0으로 입력하세요. 예: `310040275`")

        row_texts = []
        for row in range(9):
            row_texts.append(
                st.text_input(
                    f"{row + 1}행",
                    value="",
                    max_chars=9,
                    key=f"manual_row_{row}",
                    placeholder="예: 310040275"
                )
            )

        if st.button("🔎 검증 후 보관함에 저장", type="primary", key="manual_type_save"):
            board, parse_error = parse_row_strings(row_texts)

            if parse_error:
                st.error(parse_error)
            else:
                error_message, solution = validate_manual_board(board)
                if error_message:
                    st.error(error_message)
                else:
                    try:
                        save_puzzle(manual_difficulty, board, solution, source="직접입력")
                        st.success("검증을 통과했고, 보관함에 저장했습니다.")
                        st.markdown(render_board(board, solution), unsafe_allow_html=True)
                    except Exception as error:
                        st.warning(f"저장에 실패했습니다: {error}")

        if st.button("↩️ 입력값 초기화", key="manual_type_reset"):
            for row in range(9):
                st.session_state[f"manual_row_{row}"] = ""
            st.rerun()

    # -------------------------------------------------------------------
    # 방식 2: 사진으로 인식 (교정도 한 줄에 9자리씩, 9줄)
    # -------------------------------------------------------------------
    else:
        st.write("아직 안 푼 상태(인쇄된 숫자만 있는) 스도쿠 사진을 올려주세요.")

        photo_upload = st.file_uploader(
            "스도쿠 문제 사진을 업로드하세요",
            type=["jpg", "jpeg", "png"],
            key="manual_photo_upload"
        )

        if photo_upload is not None:
            file_hash = hashlib.sha256(photo_upload.getvalue()).hexdigest()

            if st.session_state.get("manual_photo_hash") != file_hash:
                st.session_state["manual_photo_hash"] = file_hash
                st.session_state["manual_photo_rotate"] = 0
                for key in ["manual_photo_crop_hash", "manual_ai_grid"]:
                    st.session_state.pop(key, None)

            try:
                manual_raw = normalize(Image.open(photo_upload))
            except Exception:
                st.error("이미지를 열 수 없습니다.")
                st.stop()

            st.session_state.setdefault("manual_photo_rotate", 0)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 90° 회전", key="manual_photo_rotate_btn"):
                    st.session_state["manual_photo_rotate"] = (st.session_state["manual_photo_rotate"] - 90) % 360
                    st.session_state.pop("manual_ai_grid", None)
            with col2:
                if st.button("↩️ 방향 초기화", key="manual_photo_reset_btn"):
                    st.session_state["manual_photo_rotate"] = 0
                    st.session_state.pop("manual_ai_grid", None)

            manual_work = resize_image(manual_raw)
            if st.session_state["manual_photo_rotate"]:
                manual_work = manual_work.rotate(st.session_state["manual_photo_rotate"], expand=True)

            manual_source_img = io.BytesIO()
            manual_work.save(manual_source_img, format="PNG")

            manual_cropped_bytes = st_cropperjs(
                pic=manual_source_img.getvalue(),
                btn_text="✅ 이 영역으로 확정",
                key="manual_photo_cropper"
            )

            manual_target = normalize(Image.open(io.BytesIO(manual_cropped_bytes))) if manual_cropped_bytes else None

            if manual_target is not None:
                crop_hash = image_hash(manual_target)
                if st.session_state.get("manual_photo_crop_hash") != crop_hash:
                    st.session_state["manual_photo_crop_hash"] = crop_hash
                    st.session_state.pop("manual_ai_grid", None)

                st.image(manual_target, caption="AI가 읽을 최종 영역", use_container_width=True)

                api_key = api_key_value()
                if not api_key:
                    st.info("Streamlit Secrets에 GEMINI_API_KEY를 설정해 주세요.")
                elif st.button("🔎 사진에서 문제 읽어오기", type="primary", key="manual_photo_read"):
                    try:
                        with st.spinner("사진 속 스도쿠 문제를 읽고 있습니다..."):
                            manual_result = ai_read_puzzle_only(get_client(api_key), manual_target, DEFAULT_MODEL)
                        st.session_state["manual_ai_grid"] = manual_result.grid
                    except Exception as error:
                        st.error("AI 판독에 실패했습니다.")
                        st.exception(error)

        if "manual_ai_grid" in st.session_state:
            st.markdown("---")
            st.markdown("### 🔧 AI가 읽은 결과를 확인하고 오류를 고쳐주세요")
            st.caption("AI가 잘못 읽은 줄이 있으면 그 줄만 수정한 뒤 저장하세요. 한 줄 9자리, 빈칸은 0입니다.")

            ai_grid = st.session_state["manual_ai_grid"]

            corrected_row_texts = []
            for row in range(9):
                default_text = "".join(str(digit) for digit in ai_grid[row])
                corrected_row_texts.append(
                    st.text_input(
                        f"{row + 1}행",
                        value=default_text,
                        max_chars=9,
                        key=f"manual_photo_row_{row}"
                    )
                )

            if st.button("🔎 검증 후 보관함에 저장", type="primary", key="manual_photo_save"):
                board, parse_error = parse_row_strings(corrected_row_texts)

                if parse_error:
                    st.error(parse_error)
                else:
                    error_message, solution = validate_manual_board(board)
                    if error_message:
                        st.error(error_message)
                    else:
                        try:
                            save_puzzle(manual_difficulty, board, solution, source="사진인식")
                            st.success("검증을 통과했고, 보관함에 저장했습니다.")
                            st.markdown(render_board(board, solution), unsafe_allow_html=True)
                            st.session_state.pop("manual_ai_grid", None)
                        except Exception as error:
                            st.warning(f"저장에 실패했습니다: {error}")
                            
# =============================================================================
# 탭 4: 문제 풀이 (보관함 → 인쇄 → 채점 → 정답 표시)
# =============================================================================
with tab_solve:
    st.subheader("📝 스도쿠 문제 풀이")

    solve_filter = st.radio(
        "조회 난이도", ["전체", "초급", "중급", "고급"],
        horizontal=True, key="solve_filter"
    )
    hide_solved = st.checkbox("✅ 이미 푼 문제 숨기기", key="solve_hide_solved")

    solve_items = load_puzzles(None if solve_filter == "전체" else solve_filter)
    if hide_solved:
        solve_items = [item for item in solve_items if not item.get("solved")]

    if not solve_items:
        st.info("풀 수 있는 문제가 보관함에 없습니다. 먼저 문제를 생성하거나 직접 입력해 주세요.")
    else:
        solve_index = st.selectbox(
            "풀 문제 선택",
            range(len(solve_items)),
            format_func=lambda i: archive_label(solve_items[i]),
            key="solve_select"
        )

        solve_item = solve_items[solve_index]
        solve_puzzle = solve_item["puzzle"]
        solve_solution = solve_item.get("solution")

        if not solve_solution:
            solve_solution = [row[:] for row in solve_puzzle]
            solve(solve_solution)

        if solve_item.get("solved"):
            st.success(f"✅ 이 문제는 이미 정답을 맞혔습니다. (풀이 완료: {solve_item['solved']})")

        st.markdown("### 1. 먼저 인쇄해서 풀어보세요")
        download_buttons(solve_puzzle, solve_item.get("difficulty", "초급"), f"solve_{solve_item.get('id', solve_index)}")

        st.markdown("---")
        st.markdown("### 2. 다 풀었으면 아래에 답을 입력하세요")
        st.caption("문제에 원래 적혀 있던 숫자는 회색으로 고정되어 있고, 빈칸만 입력하면 됩니다.")

        answer_cells = [[0] * 9 for _ in range(9)]
        widget_prefix = f"solve_{solve_item.get('id', solve_index)}"
        for row in range(9):
            cols = st.columns(9)
            for col in range(9):
                fixed_value = solve_puzzle[row][col]
                if fixed_value != 0:
                    cols[col].markdown(
                        f"<div style='text-align:center;font-weight:700;color:#6b7280;"
                        f"border:1px solid #e5e7eb;border-radius:4px;padding:6px 0;'>{fixed_value}</div>",
                        unsafe_allow_html=True
                    )
                    answer_cells[row][col] = fixed_value
                else:
                    answer_cells[row][col] = cols[col].number_input(
                        label=" ",
                        min_value=0,
                        max_value=9,
                        value=0,
                        step=1,
                        key=f"{widget_prefix}_{row}_{col}",
                        label_visibility="collapsed",
                    )

        if st.button("✅ 채점하기", type="primary", key=f"{widget_prefix}_check"):
            user_board = parse_manual_board(answer_cells)
            wrong_cells = set()
            empty_cells = 0

            for row in range(9):
                for col in range(9):
                    if solve_puzzle[row][col] != 0:
                        continue
                    if user_board[row][col] == 0:
                        empty_cells += 1
                    elif user_board[row][col] != solve_solution[row][col]:
                        wrong_cells.add((row, col))

            st.markdown("### 결과")

            if empty_cells > 0:
                st.warning(f"아직 채우지 않은 칸이 {empty_cells}개 있습니다.")

            if wrong_cells:
                st.error(f"틀린 칸이 {len(wrong_cells)}개 있습니다.")
                st.markdown(render_board(user_board, errors=wrong_cells), unsafe_allow_html=True)
            elif empty_cells == 0:
                st.success("🎉 정답입니다! 모든 칸이 맞았습니다.")
                st.balloons()
                if mark_puzzle_solved(solve_item.get("id")):
                    st.info("보관함에 '풀이 완료'로 기록했습니다.")
                else:
                    st.warning("정답은 맞았지만, 보관함 기록에는 실패했습니다.")
            else:
                st.info("지금까지 입력한 부분은 모두 맞았습니다. 나머지 칸을 마저 채워보세요.")
