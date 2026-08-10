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
# 기본 설정 및 비밀번호
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

    st.text_input(
        "비밀번호를 입력하세요",
        type="password",
        on_change=password_entered,
        key="password",
    )

    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("비밀번호가 틀렸습니다.")

    return False


if not check_password():
    st.stop()


st.markdown("""
    <style>
        .stApp { max-width: 100%; padding-left: 0.5rem; padding-right: 0.5rem; }
        .app-title { text-align: center; margin: 0.4rem 0 1.4rem; }
        .sudoku-wrap { display: flex; justify-content: center; overflow-x: auto; margin: 15px 0; }
        .sudoku { border-collapse: collapse; border: 3px solid #222222; background: #ffffff; }
        .sudoku td {
            width: 36px; height: 36px; border: 1px solid #cccccc;
            text-align: center; vertical-align: middle;
            font-size: 18px; font-weight: 700; color: #111111;
        }
        .sudoku tr:nth-child(3n) td { border-bottom: 2px solid #222222; }
        .sudoku td:nth-child(3n) { border-right: 2px solid #222222; }
        .sudoku tr:first-child td { border-top: 2px solid #222222; }
        .sudoku td:first-child { border-left: 2px solid #222222; }
        .sudoku td.error { background: #fecaca; color: #991b1b; }
        .sudoku td.hint { background: #fef3c7; }
        .sudoku td.answer { background: #eff6ff; color: #1d4ed8; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='app-title'>🏄 영용's Sudoku</h1>", unsafe_allow_html=True)

MAX_IMAGE_DIM = 768
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")


# =============================================================================
# Gemini 응답 스키마
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


class GridOnly(BaseModel):
    grid: list[list[int]]


@st.cache_resource
def get_client(api_key):
    return genai.Client(api_key=api_key)


def api_key_value():
    return st.secrets.get("GEMINI_API_KEY", None) or os.getenv("GEMINI_API_KEY")


def validate_grid(grid):
    if len(grid) != 9 or any(len(row) != 9 for row in grid):
        raise ValueError("AI가 인식한 결과가 9x9 형태가 아닙니다.")
    if any((not isinstance(n, int) or n < 0 or n > 9) for row in grid for n in row):
        raise ValueError("숫자는 0~9 사이여야 합니다.")
    return grid


def ai_read_sudoku(client, image, model_name):
    instruction = (
        "당신은 스도쿠 이미지를 읽는 AI입니다. "
        "사진 속 9x9 스도쿠 판을 정확히 읽어 grid로 반환하세요.\n"
        "- grid는 9행 9열의 정수 배열입니다.\n"
        "- 빈칸은 0으로 표기합니다.\n"
        "- 스도쿠 규칙에 위배되는 숫자가 있다면 errors에 기록하세요.\n"
        "- 지금 바로 확실하게 채울 수 있는 칸이 있다면 single_hint로 알려주세요."
    )
    response = client.models.generate_content(
        model=model_name,
        contents=[image, "이 스도쿠 판을 읽고 JSON으로 반환하세요."],
        config=types.GenerateContentConfig(
            system_instruction=instruction,
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=SudokuAnalysis,
        ),
    )
    if not response or not response.text:
        raise RuntimeError("Gemini 응답을 받지 못했습니다.")
    result = SudokuAnalysis.model_validate_json(response.text)
    result.grid = validate_grid(result.grid)
    return result


def ai_read_puzzle_only(client, image, model_name):
    instruction = (
        "당신은 스도쿠 이미지를 읽는 AI입니다. "
        "사진 속 9x9 스도쿠 판의 숫자만 grid로 반환하세요. 빈칸은 0입니다."
    )
    response = client.models.generate_content(
        model=model_name,
        contents=[image, "이 스도쿠 판의 숫자만 읽어 JSON으로 반환하세요."],
        config=types.GenerateContentConfig(
            system_instruction=instruction,
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=GridOnly,
        ),
    )
    if not response or not response.text:
        raise RuntimeError("Gemini 응답을 받지 못했습니다.")
    result = GridOnly.model_validate_json(response.text)
    result.grid = validate_grid(result.grid)
    return result


# =============================================================================
# 이미지 유틸
# =============================================================================
def normalize_image(image):
    return ImageOps.exif_transpose(image).convert("RGB")


def resize_image(image, max_dim=MAX_IMAGE_DIM):
    image = normalize_image(image)
    width, height = image.size
    if max(width, height) <= max_dim:
        return image
    ratio = max_dim / max(width, height)
    return image.resize((int(width * ratio), int(height * ratio)), Image.Resampling.LANCZOS)


def image_hash(image):
    image = normalize_image(image)
    return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def draw_photo_x(image, cells):
    result = normalize_image(image).copy()
    draw = ImageDraw.Draw(result)
    width, height = result.size
    cell_w, cell_h = width / 9, height / 9
    line_w = max(3, int(width / 80))

    for row, col in cells:
        x1, y1 = col * cell_w + cell_w * 0.15, row * cell_h + cell_h * 0.15
        x2, y2 = (col + 1) * cell_w - cell_w * 0.15, (row + 1) * cell_h - cell_h * 0.15
        draw.line((x1, y1, x2, y2), fill="red", width=line_w)
        draw.line((x1, y2, x2, y1), fill="red", width=line_w)

    return result


# =============================================================================
# 스도쿠 생성/검증 로직
# =============================================================================
def is_valid(board, row, col, number):
    for index in range(9):
        if board[row][index] == number or board[index][col] == number:
            return False
        if board[3 * (row // 3) + index // 3][3 * (col // 3) + index % 3] == number:
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
                    return (row, col, [])
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
    base = [[(row * 3 + row // 3 + col) % 9 + 1 for col in range(9)] for row in range(9)]

    digits = list(range(1, 10))
    random.shuffle(digits)
    board = [[digits[value - 1] for value in row] for row in base]

    bands = [0, 1, 2]
    random.shuffle(bands)
    rows = []
    for band in bands:
        inner_rows = [0, 1, 2]
        random.shuffle(inner_rows)
        rows.extend(band * 3 + item for item in inner_rows)
    board = [board[row] for row in rows]

    stacks = [0, 1, 2]
    random.shuffle(stacks)
    cols = []
    for stack in stacks:
        inner_cols = [0, 1, 2]
        random.shuffle(inner_cols)
        cols.extend(stack * 3 + item for item in inner_cols)
    board = [[row[col] for col in cols] for row in board]

    if random.choice([True, False]):
        board = [[board[col][row] for col in range(9)] for row in range(9)]

    return board


def clue_count(board):
    return sum(value != 0 for row in board for value in row)


def make_rotational_groups():
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
    backup = [(row, col, puzzle[row][col]) for row, col in group]
    for row, col, _ in backup:
        puzzle[row][col] = 0

    if count_solutions([row[:] for row in puzzle], limit=2) == 1:
        return True

    for row, col, value in backup:
        puzzle[row][col] = value
    return False


def generate_puzzle(difficulty, max_board_attempts=8):
    needed = {"초급": 38, "중급": 30, "고급": 24}.get(difficulty, 30)
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

        if left <= needed:
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
            inspect([(row, col) for row in range(box_row, box_row + 3) for col in range(box_col, box_col + 3)])

    return errors


def find_hint_cells(board):
    hints = set()

    for row in range(9):
        for col in range(9):
            if board[row][col] == 0 and len(candidates(board, row, col)) == 1:
                hints.add((row, col))

    groups = []
    groups += [[(row, col) for col in range(9)] for row in range(9)]
    groups += [[(row, col) for row in range(9)] for col in range(9)]
    groups += [
        [(row, col) for row in range(br, br + 3) for col in range(bc, bc + 3)]
        for br in range(0, 9, 3) for bc in range(0, 9, 3)
    ]

    for group in groups:
        possible = {}
        for row, col in group:
            if board[row][col] == 0:
                for number in candidates(board, row, col):
                    possible.setdefault(number, []).append((row, col))
        for cells in possible.values():
            if len(cells) == 1:
                hints.add(cells[0])

    return hints


def parse_row_strings(row_texts):
    board = []
    for index, row_text in enumerate(row_texts):
        cleaned = row_text.strip()
        if len(cleaned) != 9 or not cleaned.isdigit():
            return None, f"{index + 1}행은 숫자 9자리여야 합니다. (빈칸은 0)"
        board.append([int(ch) for ch in cleaned])
    return board, None


def validate_manual_board(board):
    solution = [row[:] for row in board]
    if not solve(solution):
        return "입력한 문제는 풀 수 없는 스도쿠입니다. 숫자를 다시 확인해 주세요.", None

    check_board = [row[:] for row in board]
    if count_solutions(check_board, limit=2) != 1:
        return "입력한 문제는 해가 여러 개일 수 있습니다. 숫자를 다시 확인해 주세요.", None

    return None, solution


# =============================================================================
# 저장소 (Google Sheets)
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
                "created_at": row.get("created_at", ""),
                "source": row.get("source", ""),
                "solved": row.get("solved", ""),
            })
        except (KeyError, ValueError, json.JSONDecodeError):
            continue

    if difficulty:
        items = [item for item in items if item["difficulty"] == difficulty]

    return items


def save_puzzle(difficulty, puzzle, solution, source=""):
    worksheet = get_worksheet()
    existing = load_puzzles()
    new_id = max((item["id"] for item in existing), default=0) + 1
    created_at = dt.datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S KST")

    worksheet.append_row([
        new_id,
        difficulty,
        json.dumps(puzzle),
        json.dumps(solution),
        created_at,
        source,
        "",
    ])
    return new_id


def set_puzzle_solved_status(puzzle_id, solved_text):
    """보관함 항목의 '풀이 완료' 칸을 설정하거나(문자열) 해제(빈 문자열)한다."""
    try:
        worksheet = get_worksheet()
        cell = worksheet.find(str(puzzle_id))
        if cell is None:
            return False
        worksheet.update_cell(cell.row, SOLVED_COLUMN_INDEX, solved_text)
        return True
    except Exception:
        return False


def mark_puzzle_solved(puzzle_id):
    solved_at = dt.datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M")
    return set_puzzle_solved_status(puzzle_id, solved_at)


def unmark_puzzle_solved(puzzle_id):
    return set_puzzle_solved_status(puzzle_id, "")


def archive_label(item):
    tag = f" · {item['source']}" if item.get("source") else ""
    solved_tag = " ✅" if item.get("solved") else ""
    return f"#{item['id']} · {item['difficulty']} · {item.get('created_at', '')}{tag}{solved_tag}"


# =============================================================================
# 렌더링
# =============================================================================
def render_board(board, solution=None, errors=None, hints=None):
    errors, hints = errors or set(), hints or set()
    html = "<div class='sudoku-wrap'><table class='sudoku'>"

    for row in range(9):
        html += "<tr>"
        for col in range(9):
            value = board[row][col]
            css = (
                "error" if (row, col) in errors
                else "hint" if (row, col) in hints
                else "answer" if value == 0 and solution
                else ""
            )
            shown = value or (solution[row][col] if solution else "")
            html += f"<td class='{css}'>{shown}</td>"
        html += "</tr>"

    html += "</table></div>"
    return html


def load_font(size, bold=False):
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for path in paths:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def make_png(board, title="Daily Sudoku Puzzle"):
    cell, margin, title_h = 72, 42, 72
    size = cell * 9
    image = Image.new("RGB", (size + margin * 2, size + margin * 2 + title_h), "white")
    draw = ImageDraw.Draw(image)
    draw.text((margin, 18), title, fill="#111827", font=load_font(28, True))

    x0, y0 = margin, margin + title_h
    for index in range(10):
        width = 5 if index % 3 == 0 else 1
        draw.line((x0 + index * cell, y0, x0 + index * cell, y0 + size), fill="#111", width=width)
        draw.line((x0, y0 + index * cell, x0 + size, y0 + index * cell), fill="#111", width=width)

    number_font = load_font(38, True)
    for row in range(9):
        for col in range(9):
            if board[row][col]:
                text = str(board[row][col])
                box = draw.textbbox((0, 0), text, font=number_font)
                draw.text(
                    (x0 + col * cell + (cell - box[2]) / 2, y0 + row * cell + (cell - box[3]) / 2 - 4),
                    text, fill="#111", font=number_font,
                )

    data = io.BytesIO()
    image.save(data, format="PNG", optimize=True)
    return data.getvalue()


def make_pdf(board, date_value, difficulty):
    data = io.BytesIO()
    pdf = canvas.Canvas(data, pagesize=A4)
    width, height = A4

    size = 70 * mm
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
    square, gap = 5 * mm, 2 * mm
    start = width / 2 - (count * square + (count - 1) * gap) / 2
    y = height - 49 * mm
    pdf.setFillColor(HexColor("#2563EB"))
    for index in range(count):
        pdf.roundRect(start + index * (square + gap), y, square, square, 1.2 * mm, stroke=0, fill=1)

    pdf.setStrokeColor(HexColor("#111111"))
    for index in range(10):
        pdf.setLineWidth(1.2 if index % 3 == 0 else 0.3)
        position = index * cell
        pdf.line(left + position, bottom, left + position, bottom + size)
        pdf.line(left, bottom + position, left + size, bottom + position)

    pdf.setFillColor(HexColor("#111111"))
    pdf.setFont("Helvetica-Bold", 12)
    for row in range(9):
        for col in range(9):
            if board[row][col]:
                text = str(board[row][col])
                x = left + col * cell + (cell - stringWidth(text, "Helvetica-Bold", 12)) / 2
                y = bottom + (8 - row) * cell + cell * 0.32
                pdf.drawString(x, y, text)

    pdf.setFillColor(HexColor("#6B7280"))
    pdf.setFont("Helvetica", 9)
    pdf.drawCentredString(width / 2, 24 * mm, "Solve one square at a time. Enjoy your puzzle!")

    pdf.save()
    return data.getvalue()


def download_buttons(board, difficulty, prefix):
    st.caption("PDF 또는 PNG로 다운로드해서 인쇄하거나 사진첩에 저장할 수 있습니다.")
    date_value = st.date_input("날짜", value=dt.date.today(), key=f"{prefix}_date")
    stamp = date_value.strftime("%Y%m%d")

    left, right = st.columns(2)
    left.download_button(
        "🖨️ A4 PDF",
        make_pdf(board, date_value, difficulty),
        f"daily_sudoku_{stamp}.pdf",
        "application/pdf",
        key=f"{prefix}_pdf",
        use_container_width=True,
    )
    right.download_button(
        "🖼️ PNG",
        make_png(board, f"Daily Sudoku Puzzle - {difficulty}"),
        f"daily_sudoku_{stamp}.png",
        "image/png",
        key=f"{prefix}_png",
        use_container_width=True,
    )


# =============================================================================
# 메인 UI
# =============================================================================
tab_read, tab_create, tab_manual, tab_archive = st.tabs([
    "정답확인", "문제생성", "문제입력", "보관함"
])

# -----------------------------------------------------------------------------
# 탭 1. 정답확인 (기존 그대로)
# -----------------------------------------------------------------------------
with tab_read:
    st.subheader("1. 사진 업로드")
    upload = st.file_uploader("스도쿠 사진", type=["jpg", "jpeg", "png"])

    if upload is not None:
        file_hash = hashlib.sha256(upload.getvalue()).hexdigest()
        if st.session_state.get("upload_hash") != file_hash:
            st.session_state["upload_hash"] = file_hash
            st.session_state["rotate_angle"] = 0
            for key in ("crop_hash", "analysis", "analysis_image", "celebrated"):
                st.session_state.pop(key, None)

        try:
            raw = normalize_image(Image.open(upload))
        except Exception:
            st.error("이미지를 열 수 없습니다. 다른 파일을 업로드해 주세요.")
            st.stop()

        st.session_state.setdefault("rotate_angle", 0)

        st.subheader("2. 방향 및 영역 조정")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 90도 회전"):
                st.session_state["rotate_angle"] = (st.session_state["rotate_angle"] - 90) % 360
                st.session_state.pop("analysis", None)
        with col2:
            if st.button("↩️ 초기화"):
                st.session_state["rotate_angle"] = 0
                st.session_state.pop("analysis", None)

        work = resize_image(raw)
        if st.session_state["rotate_angle"]:
            work = work.rotate(st.session_state["rotate_angle"], expand=True)

        use_crop = st.checkbox("✂️ 9x9 영역 잘라내기", value=True, key="use_mobile_crop")
        if use_crop:
            st.info("모서리를 움직여 9x9 영역에 맞춘 뒤 완료 버튼을 누르세요.")
            source_img = io.BytesIO()
            work.save(source_img, format="PNG")
            cropped_bytes = st_cropperjs(pic=source_img.getvalue(), btn_text="완료", key="mobile_cropper")
            target = normalize_image(Image.open(io.BytesIO(cropped_bytes))) if cropped_bytes else None
        else:
            target = work

        if target is not None:
            crop_hash = image_hash(target)
            if st.session_state.get("crop_hash") != crop_hash:
                st.session_state["crop_hash"] = crop_hash
                for key in ("analysis", "analysis_image", "celebrated"):
                    st.session_state.pop(key, None)

            st.image(target, caption="AI 분석 대상 이미지", use_container_width=True)

            api_key = api_key_value()
            model = st.text_input("Gemini 모델", value=DEFAULT_MODEL, key="model_name")

            if not api_key:
                st.info("Streamlit Secrets에 GEMINI_API_KEY를 설정해 주세요.")
            elif st.button("💡 도움받기", type="primary"):
                try:
                    with st.spinner("AI가 분석 중입니다..."):
                        result = ai_read_sudoku(get_client(api_key), target, model)
                        st.session_state["analysis"] = result.model_dump()
                        st.session_state["analysis_image"] = target.copy()
                except Exception as error:
                    st.error("AI 분석 중 오류가 발생했습니다. API 키, 네트워크, 모델 이름을 확인해 주세요.")
                    st.exception(error)

            if "analysis" in st.session_state:
                grid = validate_grid(SudokuAnalysis.model_validate(st.session_state["analysis"]).grid)

                with st.expander("🔧 AI가 잘못 읽은 숫자 수정", expanded=False):
                    st.caption("사진과 대조하면서 잘못 읽은 줄만 고치세요. 한 줄 9자리, 빈칸은 0입니다.")
                    row_inputs = []
                    for r in range(9):
                        row_str = "".join(str(n) for n in grid[r])
                        label_col, input_col = st.columns([1, 5])
                        label_col.markdown(f"{r + 1}행")
                        row_inputs.append(
                            input_col.text_input(
                                f"row{r}", value=row_str, max_chars=9,
                                key=f"row_fix_{r}", label_visibility="collapsed",
                            )
                        )
                    if st.button("적용", type="primary", key="apply_row_fix"):
                        try:
                            new_grid = []
                            for r, row_str in enumerate(row_inputs):
                                cleaned = row_str.strip()
                                if len(cleaned) != 9 or not cleaned.isdigit():
                                    raise ValueError(f"{r + 1}행은 숫자 9자리여야 합니다.")
                                new_grid.append([int(ch) for ch in cleaned])
                            new_grid = validate_grid(new_grid)
                            st.session_state["analysis"]["grid"] = new_grid
                            st.session_state.pop("celebrated", None)
                            st.success("수정이 반영되었습니다.")
                            st.rerun()
                        except ValueError as error:
                            st.error(str(error))

                errors = find_rule_errors(grid)
                complete = all(number != 0 for row in grid for number in row)

                st.markdown("---")
                st.subheader("AI 분석 결과")

                if errors:
                    st.markdown(render_board(grid, errors=errors), unsafe_allow_html=True)
                    st.error("규칙에 위배되는 숫자가 있습니다.")
                    if "analysis_image" in st.session_state:
                        st.image(
                            draw_photo_x(st.session_state["analysis_image"], errors),
                            caption="빨간 X 표시 확인", use_container_width=True,
                        )
                elif complete:
                    st.markdown(render_board(grid), unsafe_allow_html=True)
                    st.success("모든 칸이 채워졌고 규칙 위반이 없습니다!")
                    finish_hash = hashlib.sha256(json.dumps(grid).encode()).hexdigest()
                    if st.session_state.get("celebrated") != finish_hash:
                        st.session_state["celebrated"] = finish_hash
                        st.balloons()
                else:
                    hints = find_hint_cells(grid)
                    st.markdown(render_board(grid, hints=hints), unsafe_allow_html=True)
                    st.info("노란색 칸은 지금 바로 채울 수 있는 칸입니다." if hints else "지금은 확실한 힌트가 없습니다.")

                st.download_button(
                    "🖼️ 인식된 판 PNG",
                    make_png(grid, "AI Read Sudoku Grid"),
                    "recognized_sudoku_grid.png",
                    "image/png",
                    key="read_png",
                    use_container_width=True,
                )

# -----------------------------------------------------------------------------
# 탭 2. 문제생성 (보관함 조회 섹션 제거, 방금 생성한 문제만 표시 + PDF/PNG 다운로드)
# -----------------------------------------------------------------------------
with tab_create:
    st.subheader("🎲 난이도별 스도쿠 문제 생성")

    first, second = st.columns([2, 1])
    with first:
        difficulty = st.selectbox("난이도", ["초급", "중급", "고급"], key="create_difficulty")
    with second:
        st.write("")
        st.write("")
        if st.button("문제 생성", type="primary", use_container_width=True):
            with st.spinner("문제를 만드는 중..."):
                puzzle, answer = generate_puzzle(difficulty)
                new_id = None
                try:
                    new_id = save_puzzle(difficulty, puzzle, answer, source="생성")
                    st.success("새 문제가 만들어져 보관함에 저장되었습니다.")
                except Exception as error:
                    st.warning(f"보관함 저장 중 오류: {error}")
                st.session_state.update(
                    puzzle=puzzle, answer=answer, difficulty=difficulty, puzzle_id=new_id,
                )

    if "puzzle" in st.session_state:
        st.markdown(f"### 📋 방금 만든 문제 ({st.session_state['difficulty']})")
        show = st.toggle("🔍 정답 보기", key="current_solution")
        st.markdown(
            render_board(
                st.session_state["puzzle"],
                st.session_state["answer"] if show else None,
            ),
            unsafe_allow_html=True,
        )

        st.markdown("#### 🖨️ 인쇄용 다운로드")
        download_buttons(st.session_state["puzzle"], st.session_state["difficulty"], "new")

# -----------------------------------------------------------------------------
# 탭 3. 문제입력 (직접 입력 / 사진 AI 입력, 방금 입력한 것만 표시 + PDF/PNG 다운로드)
# -----------------------------------------------------------------------------
with tab_manual:
    st.subheader("✍️ 문제 직접 입력")
    st.caption("사진 속 문제를 직접 입력하거나, AI가 읽어준 결과를 수정해서 저장할 수 있습니다.")

    input_method = st.radio("입력 방식", ["직접 입력", "사진으로 AI 입력"], horizontal=True, key="manual_input_method")
    manual_difficulty = st.selectbox("난이도", ["초급", "중급", "고급"], key="manual_difficulty")

    if input_method == "직접 입력":
        if st.session_state.get("manual_reset_pending"):
            for row in range(9):
                st.session_state[f"manual_row_{row}"] = ""
            st.session_state["manual_reset_pending"] = False
            st.session_state.pop("manual_saved_puzzle", None)

        st.write("한 줄에 9자리 숫자, 빈칸은 0. 예: 310040275")
        row_texts = []
        for row in range(9):
            label_col, input_col = st.columns([1, 5])
            label_col.markdown(
                f"<div style='padding-top:0.55rem;font-weight:600;'>{row + 1}행</div>",
                unsafe_allow_html=True,
            )
            row_texts.append(
                input_col.text_input(
                    "row", value="", max_chars=9,
                    key=f"manual_row_{row}", placeholder="310040275",
                    label_visibility="collapsed",
                )
            )

        preview_board, preview_error = parse_row_strings(row_texts)
        if preview_board is not None:
            st.markdown("#### 미리보기")
            preview_errors = find_rule_errors(preview_board)
            st.markdown(render_board(preview_board, errors=preview_errors), unsafe_allow_html=True)
            if preview_errors:
                st.warning("규칙에 위배되는 숫자가 있습니다.")
        elif any(text.strip() for text in row_texts):
            st.caption("9행 모두 숫자 9자리를 입력해야 미리보기가 표시됩니다.")

        if st.button("저장", type="primary", key="manual_type_save"):
            board, parse_error = parse_row_strings(row_texts)
            if parse_error:
                st.error(parse_error)
            else:
                error_message, solution = validate_manual_board(board)
                if error_message:
                    st.error(error_message)
                else:
                    try:
                        new_id = save_puzzle(manual_difficulty, board, solution, source="직접입력")
                        st.success("문제가 저장되었습니다.")
                        st.session_state["manual_saved_puzzle"] = {
                            "id": new_id,
                            "board": board,
                            "solution": solution,
                            "difficulty": manual_difficulty,
                        }
                    except Exception as error:
                        st.warning(f"저장 중 오류: {error}")

        if "manual_saved_puzzle" in st.session_state:
            saved = st.session_state["manual_saved_puzzle"]
            st.markdown(f"### 📋 방금 저장한 문제 ({saved['difficulty']})")
            show_manual_solution = st.toggle("🔍 정답 보기", key="manual_type_show_solution")
            st.markdown(
                render_board(saved["board"], saved["solution"] if show_manual_solution else None),
                unsafe_allow_html=True,
            )

            st.markdown("#### 🖨️ 인쇄용 다운로드")
            download_buttons(saved["board"], saved["difficulty"], "manual_type")

        if st.button("초기화", key="manual_type_reset"):
            st.session_state["manual_reset_pending"] = True
            st.rerun()

    else:
        st.write("스도쿠 사진을 업로드하면 AI가 숫자를 읽어줍니다.")
        photo_upload = st.file_uploader("문제 사진", type=["jpg", "jpeg", "png"], key="manual_photo_upload")

        if photo_upload is not None:
            file_hash = hashlib.sha256(photo_upload.getvalue()).hexdigest()
            if st.session_state.get("manual_photo_hash") != file_hash:
                st.session_state["manual_photo_hash"] = file_hash
                st.session_state["manual_photo_rotate"] = 0
                for key in ("manual_photo_crop_hash", "manual_ai_grid", "manual_photo_saved_puzzle"):
                    st.session_state.pop(key, None)

            try:
                manual_raw = normalize_image(Image.open(photo_upload))
            except Exception:
                st.error("이미지를 열 수 없습니다.")
                st.stop()

            st.session_state.setdefault("manual_photo_rotate", 0)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 90도 회전", key="manual_photo_rotate_btn"):
                    st.session_state["manual_photo_rotate"] = (st.session_state["manual_photo_rotate"] - 90) % 360
                    st.session_state.pop("manual_ai_grid", None)
            with col2:
                if st.button("↩️ 초기화", key="manual_photo_reset_btn"):
                    st.session_state["manual_photo_rotate"] = 0
                    st.session_state.pop("manual_ai_grid", None)

            manual_work = resize_image(manual_raw)
            if st.session_state["manual_photo_rotate"]:
                manual_work = manual_work.rotate(st.session_state["manual_photo_rotate"], expand=True)

            manual_source_img = io.BytesIO()
            manual_work.save(manual_source_img, format="PNG")
            manual_cropped_bytes = st_cropperjs(
                pic=manual_source_img.getvalue(), btn_text="완료", key="manual_photo_cropper",
            )
            manual_target = normalize_image(Image.open(io.BytesIO(manual_cropped_bytes))) if manual_cropped_bytes else None

            if manual_target is not None:
                crop_hash = image_hash(manual_target)
                if st.session_state.get("manual_photo_crop_hash") != crop_hash:
                    st.session_state["manual_photo_crop_hash"] = crop_hash
                    st.session_state.pop("manual_ai_grid", None)

                st.image(manual_target, caption="AI 분석 대상", use_container_width=True)

                api_key = api_key_value()
                if not api_key:
                    st.info("Streamlit Secrets에 GEMINI_API_KEY를 설정해 주세요.")
                elif st.button("AI로 숫자 읽기", type="primary", key="manual_photo_read"):
                    try:
                        with st.spinner("AI가 분석 중입니다..."):
                            manual_result = ai_read_puzzle_only(get_client(api_key), manual_target, DEFAULT_MODEL)
                            st.session_state["manual_ai_grid"] = manual_result.grid
                    except Exception as error:
                        st.error("AI 분석 중 오류가 발생했습니다.")
                        st.exception(error)

                if "manual_ai_grid" in st.session_state:
                    st.markdown("---")
                    st.markdown("#### AI가 읽은 결과 확인/수정")
                    st.caption("사진과 대조하면서 잘못 읽은 줄만 고치세요. 한 줄 9자리, 빈칸은 0입니다.")

                    ai_grid = st.session_state["manual_ai_grid"]
                    corrected_row_texts = []
                    for row in range(9):
                        default_text = "".join(str(digit) for digit in ai_grid[row])
                        label_col, input_col = st.columns([1, 6])
                        label_col.markdown(
                            f"<div style='padding-top:0.55rem;font-weight:600;white-space:nowrap;'>{row + 1}행</div>",
                            unsafe_allow_html=True,
                        )
                        corrected_row_texts.append(
                            input_col.text_input(
                                "row", value=default_text, max_chars=9,
                                key=f"manual_photo_row_{row}", label_visibility="collapsed",
                            )
                        )

                    preview_board, preview_error = parse_row_strings(corrected_row_texts)
                    if preview_board is not None:
                        st.markdown("#### 미리보기")
                        preview_errors = find_rule_errors(preview_board)
                        st.markdown(render_board(preview_board, errors=preview_errors), unsafe_allow_html=True)
                        if preview_errors:
                            st.warning("규칙에 위배되는 숫자가 있습니다.")
                    else:
                        st.caption("9행 모두 숫자 9자리를 입력해야 미리보기가 표시됩니다.")

                    if st.button("저장", type="primary", key="manual_photo_save"):
                        board, parse_error = parse_row_strings(corrected_row_texts)
                        if parse_error:
                            st.error(parse_error)
                        else:
                            error_message, solution = validate_manual_board(board)
                            if error_message:
                                st.error(error_message)
                            else:
                                try:
                                    new_id = save_puzzle(manual_difficulty, board, solution, source="사진입력")
                                    st.success("문제가 저장되었습니다.")
                                    st.session_state["manual_photo_saved_puzzle"] = {
                                        "id": new_id,
                                        "board": board,
                                        "solution": solution,
                                        "difficulty": manual_difficulty,
                                    }
                                    st.session_state.pop("manual_ai_grid", None)
                                except Exception as error:
                                    st.warning(f"저장 중 오류: {error}")

                if "manual_photo_saved_puzzle" in st.session_state:
                    saved = st.session_state["manual_photo_saved_puzzle"]
                    st.markdown(f"### 📋 방금 저장한 문제 ({saved['difficulty']})")
                    show_photo_solution = st.toggle("🔍 정답 보기", key="manual_photo_show_solution")
                    st.markdown(
                        render_board(saved["board"], saved["solution"] if show_photo_solution else None),
                        unsafe_allow_html=True,
                    )

                    st.markdown("#### 🖨️ 인쇄용 다운로드")
                    download_buttons(saved["board"], saved["difficulty"], "manual_photo")

# -----------------------------------------------------------------------------
# 탭 4. 보관함 (검색/조회 + 풀이 완료 체크 토글)
# -----------------------------------------------------------------------------
with tab_archive:
    st.subheader("📁 저장된 문제 보관함")
    st.caption("난이도, 풀이 완료 여부로 검색해서 저장된 문제를 확인하고, 풀이 완료 여부를 체크할 수 있습니다.")

    filter_col, hide_col = st.columns([2, 1])
    with filter_col:
        archive_filter = st.radio(
            "난이도 검색", ["전체", "초급", "중급", "고급"],
            horizontal=True, key="archive_filter",
        )
    with hide_col:
        st.write("")
        hide_solved = st.checkbox("✅ 푼 문제 숨기기", key="archive_hide_solved")

    archive_items = load_puzzles(None if archive_filter == "전체" else archive_filter)
    if hide_solved:
        archive_items = [item for item in archive_items if not item.get("solved")]

    st.caption(f"검색 결과: {len(archive_items)}개")

    if not archive_items:
        st.info("조건에 맞는 저장된 문제가 없습니다.")
    else:
        archive_index = st.selectbox(
            "문제 선택",
            range(len(archive_items)),
            format_func=lambda i: archive_label(archive_items[i]),
            key="archive_select",
        )

        archive_item = archive_items[archive_index]
        archive_id = archive_item.get("id")
        archive_puzzle = archive_item["puzzle"]
        archive_solution = archive_item.get("solution")

        if not archive_solution:
            archive_solution = [row[:] for row in archive_puzzle]
            solve(archive_solution)

        # -------------------------------------------------------------------
        # 풀이 완료 여부 체크박스
        # -------------------------------------------------------------------
        solved_key = f"archive_solved_{archive_id}"
        currently_solved = bool(archive_item.get("solved"))

        checked = st.checkbox("✅ 푼 문제로 표시", value=currently_solved, key=solved_key)

        if checked != currently_solved:
            success = mark_puzzle_solved(archive_id) if checked else unmark_puzzle_solved(archive_id)
            if success:
                st.rerun()
            else:
                st.warning("상태 변경 중 오류가 발생했습니다.")

        if archive_item.get("solved"):
            st.caption(f"완료일: {archive_item['solved']}")

        show_archive_solution = st.toggle("🔍 정답 보기", key="archive_show_solution")
        st.markdown(
            render_board(archive_puzzle, archive_solution if show_archive_solution else None),
            unsafe_allow_html=True,
        )

        st.markdown("#### 🖨️ 인쇄용 다운로드")
        download_buttons(
            archive_puzzle,
            archive_item.get("difficulty", "초급"),
            f"archive_{archive_id if archive_id is not None else archive_index}",
        )
