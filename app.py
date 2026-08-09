import os
import io
import json
import random
import hashlib
import datetime
from typing import Literal, Optional

import streamlit as st
from PIL import Image, ImageDraw, ImageOps, ImageFont
from streamlit_cropperjs import st_cropperjs
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from pydantic import BaseModel, Field
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

# ------------------------------------------------------------------------------
# 1. App configuration
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Miracle Morning SUDOKU", page_icon="🧩", layout="centered")

st.markdown(
    """
    <style>
        html { scroll-behavior: smooth; }
        .stApp { max-width: 100%; padding-left: 0.5rem; padding-right: 0.5rem; }
        iframe { max-width: 100% !important; width: 100% !important; }
        img { max-width: 100% !important; height: auto !important; }

        /* 제목이 휴대폰 화면 폭에 맞춰 자동으로 축소되어 한 줄을 유지하도록 처리 */
        .app-title {
            font-size: clamp(1.15rem, 4.5vw, 2.4rem);
            font-weight: 800;
            letter-spacing: -0.02em;
            white-space: nowrap;
            overflow-x: hidden;
            text-align: center;
            margin: 0.25rem 0 0.75rem 0;
        }

        .home-button-wrap { text-align: center; margin: 20px 0 8px 0; }
        .home-button-wrap a { text-decoration: none; }
        .home-button-wrap button {
            width: 100%;
            padding: 12px 20px;
            border-radius: 10px;
            border: 1px solid rgba(0,0,0,0.15);
            background: #f7f6f2;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
        }
        .home-button-wrap button:hover { background: #edeae5; }

        /* 9x9 판독 결과 격자 스타일링 */
        .st-key-sudoku_band_0,
        .st-key-sudoku_band_1,
        .st-key-sudoku_band_2 {
            border: 2px solid #222 !important;
            border-radius: 6px !important;
            padding: 3px !important;
            margin-bottom: 3px !important;
        }
        .st-key-sudoku_band_0 div[data-testid="stHorizontalBlock"],
        .st-key-sudoku_band_1 div[data-testid="stHorizontalBlock"],
        .st-key-sudoku_band_2 div[data-testid="stHorizontalBlock"] {
            gap: 2px !important;
        }
        .st-key-sudoku_band_0 div[data-testid="column"],
        .st-key-sudoku_band_1 div[data-testid="column"],
        .st-key-sudoku_band_2 div[data-testid="column"] {
            padding: 0 !important;
            min-width: 0 !important;
        }
        .st-key-sudoku_band_0 input,
        .st-key-sudoku_band_1 input,
        .st-key-sudoku_band_2 input {
            text-align: center !important;
            padding: 4px 0 !important;
            font-size: 17px !important;
            font-weight: 700 !important;
            height: 38px !important;
            border: 1px solid #ccc !important;
            border-radius: 4px !important;
        }
        .st-key-sudoku_band_0 div[data-testid="stHorizontalBlock"] > div:nth-of-type(3) input,
        .st-key-sudoku_band_0 div[data-testid="stHorizontalBlock"] > div:nth-of-type(6) input,
        .st-key-sudoku_band_1 div[data-testid="stHorizontalBlock"] > div:nth-of-type(3) input,
        .st-key-sudoku_band_1 div[data-testid="stHorizontalBlock"] > div:nth-of-type(6) input,
        .st-key-sudoku_band_2 div[data-testid="stHorizontalBlock"] > div:nth-of-type(3) input,
        .st-key-sudoku_band_2 div[data-testid="stHorizontalBlock"] > div:nth-of-type(6) input {
            border-right: 3px solid #222 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# gemini-2.5-* 계열은 신규 사용자에게 404(NOT_FOUND)가 발생할 수 있으므로 절대 사용하지 않는다.
MODEL_CANDIDATES = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
]

LOCAL_PUZZLE_FILE = "puzzles_db.json"
DRIVE_FILE_NAME = "puzzles_db.json"
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.file"]

# secrets.toml에 [gcp_service_account]와 DRIVE_FOLDER_ID가 모두 설정되어 있으면
# 구글 드라이브를 저장소로 사용하고, 없으면 로컬 JSON 파일로 자동 대체(fallback)한다.
DRIVE_ENABLED = bool(st.secrets.get("gcp_service_account")) and bool(st.secrets.get("DRIVE_FOLDER_ID"))

if DRIVE_ENABLED:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

    DRIVE_FOLDER_ID = st.secrets["DRIVE_FOLDER_ID"]

    @st.cache_resource
    def get_drive_service():
        info = dict(st.secrets["gcp_service_account"])
        credentials = service_account.Credentials.from_service_account_info(info, scopes=DRIVE_SCOPES)
        return build("drive", "v3", credentials=credentials, cache_discovery=False)

    def _find_drive_file_id(service):
        query = f"name='{DRIVE_FILE_NAME}' and '{DRIVE_FOLDER_ID}' in parents and trashed=false"
        results = service.files().list(q=query, spaces="drive", fields="files(id, name)").execute()
        files = results.get("files", [])
        return files[0]["id"] if files else None

    @st.cache_data(ttl=30, show_spinner=False)
    def _load_puzzles_from_drive():
        try:
            service = get_drive_service()
            file_id = _find_drive_file_id(service)
            if not file_id:
                return []
            request = service.files().get_media(fileId=file_id)
            buffer = io.BytesIO()
            downloader = MediaIoBaseDownload(buffer, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            buffer.seek(0)
            data = json.loads(buffer.read().decode("utf-8"))
            return data if isinstance(data, list) else []
        except Exception as error:
            st.warning(f"Google Drive에서 문제를 불러오지 못했습니다: {error}")
            return []

    def _save_puzzles_to_drive(puzzles):
        service = get_drive_service()
        file_id = _find_drive_file_id(service)
        content = json.dumps(puzzles, ensure_ascii=False, indent=2).encode("utf-8")
        media = MediaIoBaseUpload(io.BytesIO(content), mimetype="application/json", resumable=False)
        if file_id:
            service.files().update(fileId=file_id, media_body=media).execute()
        else:
            metadata = {"name": DRIVE_FILE_NAME, "parents": [DRIVE_FOLDER_ID]}
            service.files().create(body=metadata, media_body=media, fields="id").execute()
        _load_puzzles_from_drive.clear()


def _load_puzzles_local():
    if not os.path.exists(LOCAL_PUZZLE_FILE):
        return []
    try:
        with open(LOCAL_PUZZLE_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError, ValueError):
        return []


def _save_puzzles_local(puzzles):
    temp_file = f"{LOCAL_PUZZLE_FILE}.tmp"
    try:
        with open(temp_file, "w", encoding="utf-8") as file:
            json.dump(puzzles, file, ensure_ascii=False, indent=2)
        os.replace(temp_file, LOCAL_PUZZLE_FILE)
    except OSError as error:
        st.error(f"데이터 저장 중 오류가 발생했습니다: {error}")
        if os.path.exists(temp_file):
            os.remove(temp_file)


def load_puzzles(difficulty=None):
    puzzles = _load_puzzles_from_drive() if DRIVE_ENABLED else _load_puzzles_local()
    if difficulty:
        return [item for item in puzzles if item.get("difficulty") == difficulty]
    return puzzles


def save_puzzle(difficulty, puzzle, solution):
    puzzles = load_puzzles()
    next_id = max((item.get("id", 0) for item in puzzles), default=0) + 1
    puzzles.append(
        {
            "id": next_id,
            "difficulty": difficulty,
            "puzzle": puzzle,
            "solution": solution,
            "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    if DRIVE_ENABLED:
        try:
            _save_puzzles_to_drive(puzzles)
        except Exception as error:
            st.error(f"Google Drive 저장 중 오류가 발생했습니다: {error}")
    else:
        _save_puzzles_local(puzzles)


def render_home_button(key_suffix: str):
    st.markdown(
        '<div class="home-button-wrap"><a href="#app-top"><button type="button">'
        "🏠 처음 화면으로 돌아가기</button></a></div>",
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------------------
# 2. Gemini response schema (OCR 전용 — 정답 판정/힌트는 로컬 알고리즘이 담당)
# ------------------------------------------------------------------------------
class SudokuGridResult(BaseModel):
    grid: list[list[int]] = Field(description="9x9 정수 배열. 빈칸은 0.")
    message: str = ""


@st.cache_resource
def get_gemini_client(key: str):
    return genai.Client(api_key=key)


def get_api_key():
    key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        key = st.sidebar.text_input("Gemini API Key를 입력하세요", type="password")
    return key


# ------------------------------------------------------------------------------
# 3. Image helpers
# ------------------------------------------------------------------------------
def image_hash(image: Image.Image) -> str:
    normalized = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(normalized.size.__repr__().encode("utf-8"))
    digest.update(normalized.tobytes())
    return digest.hexdigest()


def uploaded_file_hash(uploaded_file) -> str:
    return hashlib.sha256(uploaded_file.getvalue()).hexdigest()


def resize_image(image: Image.Image, max_dim: int) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    if max(width, height) <= max_dim:
        return image
    ratio = max_dim / max(width, height)
    return image.resize(
        (max(1, int(width * ratio)), max(1, int(height * ratio))),
        Image.Resampling.LANCZOS,
    )


# ------------------------------------------------------------------------------
# 4. Sudoku algorithms (생성/검증/힌트 — 전부 로컬 계산, AI 미사용)
# ------------------------------------------------------------------------------
def is_valid(board, row, col, number):
    for index in range(9):
        if board[row][index] == number or board[index][col] == number:
            return False
        box_row = 3 * (row // 3) + index // 3
        box_col = 3 * (col // 3) + index % 3
        if board[box_row][box_col] == number:
            return False
    return True


def solve_board(board):
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                numbers = list(range(1, 10))
                random.shuffle(numbers)
                for number in numbers:
                    if is_valid(board, row, col, number):
                        board[row][col] = number
                        if solve_board(board):
                            return True
                        board[row][col] = 0
                return False
    return True


def solve_sudoku_exact(board):
    copied = [row[:] for row in board]

    def solve(current):
        for row in range(9):
            for col in range(9):
                if current[row][col] == 0:
                    for number in range(1, 10):
                        if is_valid(current, row, col, number):
                            current[row][col] = number
                            if solve(current):
                                return True
                            current[row][col] = 0
                    return False
        return True

    return copied if solve(copied) else None


def count_solutions(board, limit=2):
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                count = 0
                for number in range(1, 10):
                    if is_valid(board, row, col, number):
                        board[row][col] = number
                        count += count_solutions(board, limit - count)
                        board[row][col] = 0
                        if count >= limit:
                            return count
                return count
    return 1


def generate_sudoku_puzzle(difficulty):
    clue_target = {"초급": 38, "중급": 30, "고급": 24}.get(difficulty, 30)
    full_board = [[0] * 9 for _ in range(9)]
    solve_board(full_board)
    puzzle = [row[:] for row in full_board]

    positions = [(row, col) for row in range(9) for col in range(9)]
    random.shuffle(positions)
    remaining_clues = 81

    for row, col in positions:
        if remaining_clues <= clue_target:
            break
        previous = puzzle[row][col]
        puzzle[row][col] = 0
        if count_solutions(puzzle, limit=2) == 1:
            remaining_clues -= 1
        else:
            puzzle[row][col] = previous

    return puzzle, full_board


def find_candidates(board, row, col):
    candidates = set(range(1, 10))
    for i in range(9):
        candidates.discard(board[row][i])
        candidates.discard(board[i][col])
    box_row, box_col = 3 * (row // 3), 3 * (col // 3)
    for r in range(box_row, box_row + 3):
        for c in range(box_col, box_col + 3):
            candidates.discard(board[r][c])
    return candidates


def find_rule_violations(board):
    conflict_cells = set()

    for r in range(9):
        seen = {}
        for c in range(9):
            value = board[r][c]
            if value:
                seen.setdefault(value, []).append(c)
        for value, cols in seen.items():
            if len(cols) > 1:
                conflict_cells.update((r, c) for c in cols)

    for c in range(9):
        seen = {}
        for r in range(9):
            value = board[r][c]
            if value:
                seen.setdefault(value, []).append(r)
        for value, rows in seen.items():
            if len(rows) > 1:
                conflict_cells.update((r, c) for r in rows)

    for box_row in range(0, 9, 3):
        for box_col in range(0, 9, 3):
            seen = {}
            for r in range(box_row, box_row + 3):
                for c in range(box_col, box_col + 3):
                    value = board[r][c]
                    if value:
                        seen.setdefault(value, []).append((r, c))
            for value, cells in seen.items():
                if len(cells) > 1:
                    conflict_cells.update(cells)

    violations = [
        {
            "row": r + 1,
            "col": c + 1,
            "reason": f"숫자 {board[r][c]}가 같은 행·열·3x3 박스 안에서 중복됩니다.",
        }
        for r, c in conflict_cells
    ]
    return sorted(violations, key=lambda item: (item["row"], item["col"]))


def find_naked_single_hint(board):
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                candidates = find_candidates(board, r, c)
                if len(candidates) == 1:
                    value = next(iter(candidates))
                    return {
                        "row": r + 1,
                        "col": c + 1,
                        "number": value,
                        "reason": f"{r + 1}행 {c + 1}열은 같은 행·열·3x3 박스 규칙상 {value}만 들어갈 수 있습니다.",
                        "certain": True,
                    }
    return None


def find_fallback_hint(board):
    solved = solve_sudoku_exact(board)
    if not solved:
        return None
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                return {
                    "row": r + 1,
                    "col": c + 1,
                    "number": solved[r][c],
                    "reason": "지금 상태만으로는 값이 유일하게 결정되지 않지만, 이 값을 넣으면 끝까지 풀 수 있는 예시입니다.",
                    "certain": False,
                }
    return None


# ------------------------------------------------------------------------------
# 5. Board rendering and downloads
# ------------------------------------------------------------------------------
def render_sudoku_board_html(puzzle, solution=None):
    html = """
    <style>
        .sudoku-container { display:flex; justify-content:center; margin:15px 0; }
        .sudoku-board { border-collapse:collapse; border:3px solid #222; background:#fff; }
        .sudoku-board td { width:36px; height:36px; text-align:center; vertical-align:middle;
                            border:1px solid #ccc; font-size:18px; font-weight:bold; color:#111; }
        .sudoku-board td.solution-cell { color:#1d4ed8; background:#eff6ff; }
        .sudoku-board tr:nth-child(3n) td { border-bottom:2px solid #222; }
        .sudoku-board td:nth-child(3n) { border-right:2px solid #222; }
        .sudoku-board tr:first-child td { border-top:2px solid #222; }
        .sudoku-board td:first-child { border-left:2px solid #222; }
    </style>
    <div class="sudoku-container"><table class="sudoku-board">
    """
    for row in range(9):
        html += "<tr>"
        for col in range(9):
            value = puzzle[row][col]
            if value:
                html += f"<td>{value}</td>"
            elif solution and solution[row][col]:
                html += f'<td class="solution-cell">{solution[row][col]}</td>'
            else:
                html += "<td></td>"
        html += "</tr>"
    return html + "</table></div>"


def load_font(size, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_sudoku_board_image(puzzle, solution=None, difficulty=""):
    cell = 64
    margin = 24
    title_height = 54 if difficulty else 0
    board_size = cell * 9
    image = Image.new("RGB", (board_size + margin * 2, board_size + margin * 2 + title_height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(28, bold=True)
    number_font = load_font(32, bold=True)

    if difficulty:
        title = f"Sudoku Puzzle · {difficulty}"
        bbox = draw.textbbox((0, 0), title, font=title_font)
        draw.text(((image.width - (bbox[2] - bbox[0])) / 2, 10), title, fill="black", font=title_font)

    x0, y0 = margin, margin + title_height
    for index in range(10):
        width = 4 if index % 3 == 0 else 1
        x = x0 + index * cell
        y = y0 + index * cell
        draw.line((x, y0, x, y0 + board_size), fill="black", width=width)
        draw.line((x0, y, x0 + board_size, y), fill="black", width=width)

    for row in range(9):
        for col in range(9):
            value = puzzle[row][col]
            color = "#111111"
            if not value and solution:
                value = solution[row][col]
                color = "#1d4ed8"
            if value:
                text = str(value)
                bbox = draw.textbbox((0, 0), text, font=number_font)
                text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                draw.text(
                    (x0 + col * cell + (cell - text_w) / 2, y0 + row * cell + (cell - text_h) / 2 - 4),
                    text,
                    fill=color,
                    font=number_font,
                )
    return image


def image_to_png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def generate_sudoku_print_pdf(puzzle, print_date, difficulty=""):
    buffer = io.BytesIO()
    page_width, page_height = A4
    pdf = canvas.Canvas(buffer, pagesize=A4)
    pdf.setTitle("Daily Sudoku Puzzle")

    pdf.setFont("Helvetica-Bold", 26)
    pdf.drawCentredString(page_width / 2, page_height - 30 * mm, "Daily Sudoku Puzzle")

    date_text = print_date.strftime("%B %d, %Y")
    if difficulty:
        date_text = f"{date_text}   |   Level: {difficulty}"
    pdf.setFont("Helvetica", 12)
    pdf.drawCentredString(page_width / 2, page_height - 39 * mm, date_text)

    board_size = 160 * mm
    cell = board_size / 9
    board_x = (page_width - board_size) / 2
    board_y = (page_height - board_size) / 2 - 8 * mm

    pdf.setFillColorRGB(1, 1, 1)
    pdf.rect(board_x, board_y, board_size, board_size, stroke=0, fill=1)
    pdf.setStrokeColorRGB(0, 0, 0)

    for index in range(10):
        pdf.setLineWidth(1.7 if index % 3 == 0 else 0.35)
        pos = index * cell
        pdf.line(board_x + pos, board_y, board_x + pos, board_y + board_size)
        pdf.line(board_x, board_y + pos, board_x + board_size, board_y + pos)

    pdf.setFillColorRGB(0, 0, 0)
    pdf.setFont("Helvetica-Bold", 20)
    for row in range(9):
        for col in range(9):
            value = puzzle[row][col]
            if value:
                text = str(value)
                x = board_x + col * cell + (cell - stringWidth(text, "Helvetica-Bold", 20)) / 2
                y = board_y + board_size - (row + 1) * cell + (cell - 20) / 2 + 4
                pdf.drawString(x, y, text)

    pdf.setFont("Helvetica", 9)
    pdf.setFillColorRGB(0.35, 0.35, 0.35)
    pdf.drawCentredString(page_width / 2, 20 * mm, "Fill each row, column, and 3x3 box with the numbers 1-9.")
    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()


# ------------------------------------------------------------------------------
# 6. Gemini OCR (숫자 판독 전용 — 정답/힌트 로직은 로컬 알고리즘이 담당)
# ------------------------------------------------------------------------------
SYSTEM_PROMPT_OCR = """
당신은 이미지 속 9x9 스도쿠 판의 숫자를 정확히 판독하는 OCR 전문가입니다.
인쇄체 숫자와 손글씨 숫자를 모두 인식합니다.
숫자가 없거나 흐릿해서 판단할 수 없는 칸은 반드시 0으로 표시합니다.
추측하지 말고, 명확히 보이는 숫자만 인식하세요.

반드시 9개의 행으로 구성되고, 각 행은 9개의 정수(0~9)로 구성된 grid만 반환하세요.
"""


def ocr_sudoku_grid(client, image):
    last_error = None

    for model_name in MODEL_CANDIDATES:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    image,
                    "이 이미지 속 9x9 스도쿠 판의 숫자를 정확히 읽어 grid로 반환하세요. 빈칸은 0입니다.",
                ],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT_OCR,
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=SudokuGridResult,
                    max_output_tokens=400,
                ),
            )
            if not response or not response.text:
                raise ValueError("Gemini가 빈 응답을 반환했습니다.")

            parsed = SudokuGridResult.model_validate_json(response.text)
            fixed_grid = [[0] * 9 for _ in range(9)]
            for r in range(min(9, len(parsed.grid))):
                row = parsed.grid[r]
                for c in range(min(9, len(row))):
                    value = row[c]
                    fixed_grid[r][c] = value if isinstance(value, int) and 0 <= value <= 9 else 0
            return fixed_grid, model_name

        except genai_errors.ClientError as error:
            last_error = error
            if getattr(error, "code", None) == 404 or "NOT_FOUND" in str(error):
                continue
            raise
        except Exception as error:
            last_error = error
            continue

    raise RuntimeError(f"판독 가능한 Gemini 모델을 찾지 못했습니다: {last_error}")


def clear_ocr_state():
    for key in ("ocr_grid", "ocr_result", "ocr_used_model"):
        st.session_state.pop(key, None)


def parse_cell_value(raw: str) -> int:
    raw = (raw or "").strip()
    return int(raw) if raw in "123456789" else 0


# ------------------------------------------------------------------------------
# 7. Main UI
# ------------------------------------------------------------------------------
st.markdown('<div id="app-top"></div>', unsafe_allow_html=True)
st.markdown('<h1 class="app-title">🏄 Miracle Morning SUDOKU by Y.Y</h1>', unsafe_allow_html=True)

st.sidebar.caption(
    "☁️ 구글 드라이브에 저장 중입니다." if DRIVE_ENABLED
    else "💾 로컬 파일에 저장 중입니다. (서버 재배포 시 초기화될 수 있어요)"
)

api_key = get_api_key()
if not api_key:
    st.info("👈 사이드바 또는 Streamlit Secrets에 GEMINI_API_KEY를 설정해 주세요.")
    st.stop()
client = get_gemini_client(api_key)

tab1, tab2 = st.tabs(["📸 이미지 업로드 & 도움받기", "🎲 문제 만들기 & 보관함"])

# ==============================================================================
# TAB 1
# ==============================================================================
with tab1:
    st.subheader("1. 스도쿠 이미지 가져오기")
    img_file = st.file_uploader("스도쿠 이미지를 촬영하거나 업로드하세요", type=["jpg", "jpeg", "png"])

    if img_file is not None:
        current_upload_hash = uploaded_file_hash(img_file)
        if st.session_state.get("last_uploaded_hash") != current_upload_hash:
            st.session_state["last_uploaded_hash"] = current_upload_hash
            st.session_state.pop("last_crop_hash", None)
            clear_ocr_state()

        st.subheader("2. 9×9 영역에 맞추기")
        st.caption(
            "📱 가운데 고정된 사각 박스 안에 스도쿠 판이 들어오도록, "
            "손가락으로 사진을 확대·축소하고 움직여서 맞춘 뒤 잘라내기 버튼을 눌러주세요."
        )

        cropped_bytes = st_cropperjs(
            pic=img_file.getvalue(),
            btn_text="✅ 이 영역으로 잘라내기",
            key="mobile_cropperjs",
        )

        if cropped_bytes:
            try:
                target_img = Image.open(io.BytesIO(cropped_bytes)).convert("RGB")
                target_img = resize_image(target_img, max_dim=768)
            except (OSError, ValueError) as error:
                st.error(f"잘라낸 이미지를 불러올 수 없습니다: {error}")
                target_img = None

            if target_img is not None:
                crop_hash = image_hash(target_img)
                if st.session_state.get("last_crop_hash") != crop_hash:
                    st.session_state["last_crop_hash"] = crop_hash
                    clear_ocr_state()

                st.image(target_img, caption="최종 분석 영역", use_container_width=True)

                if st.button("🔎 손글씨 판독하기", type="primary", use_container_width=True):
                    with st.spinner("Gemini AI가 스도쿠 숫자를 읽고 있습니다..."):
                        try:
                            grid, used_model = ocr_sudoku_grid(client, target_img)
                            st.session_state["ocr_grid"] = grid
                            st.session_state["ocr_used_model"] = used_model
                            st.session_state["ocr_run_id"] = st.session_state.get("ocr_run_id", 0) + 1
                            st.session_state.pop("ocr_result", None)
                        except Exception as error:
                            st.error(f"판독 중 오류가 발생했습니다: {error}")

                if "ocr_grid" in st.session_state:
                    st.markdown("---")
                    st.subheader("3. AI 판독 결과 확인 및 수정")
                    st.caption(
                        "AI가 읽은 숫자가 아래 9×9 판에 채워져 있습니다. "
                        "잘못 읽힌 칸을 눌러 숫자를 고치거나, 빈칸으로 두려면 지워주세요."
                    )

                    run_id = st.session_state.get("ocr_run_id", 0)
                    ocr_grid = st.session_state["ocr_grid"]
                    edited_grid = [[0] * 9 for _ in range(9)]

                    for band_index, (row_start, row_end) in enumerate([(0, 3), (3, 6), (6, 9)]):
                        with st.container(border=True, key=f"sudoku_band_{band_index}"):
                            for r in range(row_start, row_end):
                                cols = st.columns(9, gap="small")
                                for c in range(9):
                                    default_value = ocr_grid[r][c]
                                    default_str = str(default_value) if default_value else ""
                                    with cols[c]:
                                        raw_value = st.text_input(
                                            f"{r + 1}행 {c + 1}열",
                                            value=default_str,
                                            max_chars=1,
                                            key=f"cell_{run_id}_{r}_{c}",
                                            label_visibility="collapsed",
                                        )
                                    edited_grid[r][c] = parse_cell_value(raw_value)

                    used_model = st.session_state.get("ocr_used_model")
                    if used_model:
                        st.caption(f"판독에 사용된 모델: {used_model}")

                    action_col1, action_col2 = st.columns(2)
                    with action_col1:
                        confirmed = st.button(
                            "✅ 이 상태로 정답확인 / 도움받기", type="primary", use_container_width=True
                        )
                    with action_col2:
                        if st.button("🔁 판독 다시 하기", use_container_width=True):
                            clear_ocr_state()
                            st.rerun()

                    if confirmed:
                        violations = find_rule_violations(edited_grid)
                        is_complete = all(
                            edited_grid[r][c] != 0 for r in range(9) for c in range(9)
                        )
                        st.session_state["ocr_result"] = {
                            "grid": edited_grid,
                            "violations": violations,
                            "is_complete": is_complete,
                        }

                    if "ocr_result" in st.session_state:
                        result = st.session_state["ocr_result"]
                        grid = result["grid"]
                        violations = result["violations"]
                        is_complete = result["is_complete"]

                        st.markdown("---")
                        st.subheader("4. 결과")

                        if violations:
                            st.error(f"⚠️ 확인 결과, 규칙에 어긋나는 칸이 {len(violations)}곳 있습니다.")
                            for violation in violations:
                                st.write(
                                    f"- **{violation['row']}행 {violation['col']}열**: {violation['reason']}"
                                )
                            st.caption("표에서 잘못 읽힌 숫자가 없는지 다시 한번 확인해 보세요.")

                        elif is_complete:
                            st.balloons()
                            st.success("🎉 정답입니다! 스도쿠를 정확히 완성했습니다.")

                        else:
                            st.success("✅ 현재까지 입력된 숫자에는 규칙 위반이 없습니다.")

                            hint = find_naked_single_hint(grid)
                            if not hint:
                                hint = find_fallback_hint(grid)

                            if hint:
                                st.subheader("💡 힌트")
                                if hint["certain"]:
                                    st.info(
                                        f"👉 **위치:** {hint['row']}행 {hint['col']}열\n\n"
                                        f"👉 **넣을 숫자:** {hint['number']}\n\n"
                                        f"👉 **풀이 이유:** {hint['reason']}"
                                    )
                                else:
                                    st.warning(
                                        f"👉 **위치:** {hint['row']}행 {hint['col']}열\n\n"
                                        f"👉 **예시 값:** {hint['number']}\n\n"
                                        f"👉 {hint['reason']}"
                                    )
                            else:
                                st.warning(
                                    "현재 입력 상태로는 힌트를 계산할 수 없습니다. "
                                    "일부 숫자가 잘못 판독되었을 수 있으니 표를 다시 확인해 주세요."
                                )

                        render_home_button("tab1_result")
        else:
            st.info("사진을 확대·축소·이동해 스도쿠 9×9 영역에 맞춘 뒤, 위 버튼으로 잘라내기를 완료하세요.")

# ==============================================================================
# TAB 2 — 문제 만들기 & 보관함 (구글 드라이브 또는 로컬 파일 기반)
# ==============================================================================
with tab2:
    st.subheader("🎲 난이도별 스도쿠 문제 생성")
    difficulty_col, button_col = st.columns([2, 1])
    with difficulty_col:
        difficulty = st.selectbox("난이도를 선택하세요", ["초급", "중급", "고급"])
    with button_col:
        st.write("")
        st.write("")
        generate_button = st.button("문제 생성", type="primary")

    if generate_button:
        with st.spinner("유일한 정답을 가진 문제를 만들고 있습니다..."):
            new_puzzle, new_solution = generate_sudoku_puzzle(difficulty)
        save_puzzle(difficulty, new_puzzle, new_solution)
        st.session_state["current_puzzle"] = new_puzzle
        st.session_state["current_solution"] = new_solution
        st.session_state["current_diff"] = difficulty
        st.success(f"새로운 {difficulty} 문제가 생성되어 보관함에 저장되었습니다!")

    if "current_puzzle" not in st.session_state:
        auto_puzzle, auto_solution = generate_sudoku_puzzle("중급")
        st.session_state["current_puzzle"] = auto_puzzle
        st.session_state["current_solution"] = auto_solution
        st.session_state["current_diff"] = "중급"

    puzzle = st.session_state["current_puzzle"]
    solution = st.session_state["current_solution"]
    current_difficulty = st.session_state["current_diff"]

    st.write(f"### 📋 생성된 문제 ({current_difficulty})")
    show_solution = st.toggle("🔍 정답 보기 (파란색 빈칸 채우기)", key="gen_sol_toggle")
    st.markdown(render_sudoku_board_html(puzzle, solution if show_solution else None), unsafe_allow_html=True)

    st.write("#### 📥 이미지로 저장")
    png_col1, png_col2 = st.columns(2)
    with png_col1:
        st.download_button(
            "📥 문제 PNG 다운로드",
            data=image_to_png_bytes(render_sudoku_board_image(puzzle, difficulty=current_difficulty)),
            file_name="sudoku_puzzle.png",
            mime="image/png",
            use_container_width=True,
            key="gen_png_puzzle",
        )
    with png_col2:
        st.download_button(
            "📥 정답 PNG 다운로드",
            data=image_to_png_bytes(render_sudoku_board_image(puzzle, solution, current_difficulty)),
            file_name="sudoku_solution.png",
            mime="image/png",
            use_container_width=True,
            key="gen_png_solution",
        )

    st.markdown("#### 🖨️ 인쇄용 PDF (A4, Daily Sudoku Puzzle)")
    print_date = st.date_input("인쇄할 날짜", value=datetime.date.today(), key="current_print_date")
    pdf_bytes = generate_sudoku_print_pdf(puzzle, print_date, current_difficulty)
    st.download_button(
        label="🖨️ A4 인쇄용 PDF 다운로드",
        data=pdf_bytes,
        file_name=f"daily_sudoku_{print_date.strftime('%Y%m%d')}.pdf",
        mime="application/pdf",
        type="primary",
        use_container_width=True,
        key="gen_pdf_download",
    )

    render_home_button("tab2_generated")

    st.markdown("---")
    st.subheader("📁 저장된 문제 보관함")
    filter_difficulty = st.radio("조회할 난이도 선택", ["전체", "초급", "중급", "고급"], horizontal=True)
    selected_difficulty = None if filter_difficulty == "전체" else filter_difficulty
    saved_puzzles = load_puzzles(selected_difficulty)

    if saved_puzzles:
        st.write(f"총 **{len(saved_puzzles)}개**의 저장된 문제가 있습니다.")
        selected_index = st.selectbox(
            "불러올 문제를 선택하세요",
            options=list(range(len(saved_puzzles))),
            format_func=lambda index: (
                f"#{saved_puzzles[index].get('id', '?')} "
                f"[{saved_puzzles[index].get('difficulty', '')}] "
                f"({saved_puzzles[index].get('created_at', '')})"
            ),
        )
        saved = saved_puzzles[selected_index]
        saved_puzzle = saved["puzzle"]
        saved_solution = saved.get("solution") or solve_sudoku_exact(saved_puzzle)
        saved_difficulty = saved.get("difficulty", "")

        show_saved_solution = st.toggle("🔍 저장된 문제 정답 보기", key="saved_sol_toggle")
        st.markdown(
            render_sudoku_board_html(saved_puzzle, saved_solution if show_saved_solution else None),
            unsafe_allow_html=True,
        )

        st.write("#### 📥 이미지로 저장")
        saved_png_col1, saved_png_col2 = st.columns(2)
        with saved_png_col1:
            st.download_button(
                "📥 문제 PNG 다운로드",
                data=image_to_png_bytes(render_sudoku_board_image(saved_puzzle, difficulty=saved_difficulty)),
                file_name=f"sudoku_puzzle_{saved.get('id', '')}.png",
                mime="image/png",
                use_container_width=True,
                key="saved_png_puzzle",
            )
        with saved_png_col2:
            if saved_solution:
                st.download_button(
                    "📥 정답 PNG 다운로드",
                    data=image_to_png_bytes(render_sudoku_board_image(saved_puzzle, saved_solution, saved_difficulty)),
                    file_name=f"sudoku_solution_{saved.get('id', '')}.png",
                    mime="image/png",
                    use_container_width=True,
                    key="saved_png_solution",
                )

        st.markdown("#### 🖨️ 인쇄용 PDF (A4, Daily Sudoku Puzzle)")
        saved_print_date = st.date_input("인쇄할 날짜", value=datetime.date.today(), key="saved_print_date")
        saved_pdf_bytes = generate_sudoku_print_pdf(saved_puzzle, saved_print_date, saved_difficulty)
        st.download_button(
            label="🖨️ 저장된 문제 A4 인쇄용 PDF 다운로드",
            data=saved_pdf_bytes,
            file_name=f"daily_sudoku_{saved_print_date.strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True,
            key="saved_pdf_download",
        )

        render_home_button("tab2_saved")
    else:
        st.info("선택한 난이도에 저장된 스도쿠 문제가 없습니다. 위에서 새 문제를 만들어 보세요!")
