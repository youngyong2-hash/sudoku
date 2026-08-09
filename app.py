import os
import io
import json
import random
import hashlib
import datetime as dt
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError
from streamlit_cropper import st_cropper
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

# -----------------------------------------------------------------------------
# App settings
# -----------------------------------------------------------------------------
st.set_page_config(page_title="🏄영용's Sudoku", page_icon="🏄", layout="centered")

st.markdown(
    """
    <style>
        .stApp { max-width: 100%; padding-left: 0.5rem; padding-right: 0.5rem; }
        .app-title { text-align: center; margin: 0.4rem 0 1.4rem; }
        iframe { max-width: 100% !important; width: 100% !important; }
        img { max-width: 100% !important; height: auto !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 class='app-title'>🏄영용's Sudoku</h1>", unsafe_allow_html=True)

PUZZLE_FILE = Path("puzzles_db.json")
DISPLAY_MAX_DIM = 350
API_MAX_DIM = 768
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")


# -----------------------------------------------------------------------------
# Gemini setup and analysis schema
# -----------------------------------------------------------------------------
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
    errors: list[SudokuError] = Field(default_factory=list)
    single_hint: SudokuHint | None = None


@st.cache_resource
def get_gemini_client(api_key: str):
    return genai.Client(api_key=api_key)


def get_api_key():
    key = st.secrets.get("GEMINI_API_KEY", None)
    return key or os.getenv("GEMINI_API_KEY")


def analyze_sudoku(client, image: Image.Image, model_name: str) -> SudokuAnalysis:
    system_prompt = """
당신은 엄격하고 명확한 스도쿠 검증 튜터입니다.
업로드된 이미지에서 9x9 스도쿠 판(인쇄체 및 손글씨)을 분석하세요.

규칙:
- errors에는 현재 적힌 숫자 중 행, 열, 3x3 박스 규칙을 위배하는 숫자만 넣으세요.
- 오류가 없으면 errors는 빈 배열입니다.
- single_hint에는 현재 판에서 논리적으로 확실하게 채울 수 있는 단 한 칸만 제시하세요.
- 확실한 힌트를 찾을 수 없으면 single_hint는 null입니다.
- 행과 열은 반드시 1부터 9까지입니다.
- 보드 밖의 숫자, 메모, 장식은 무시하세요.
"""

    response = client.models.generate_content(
        model=model_name,
        contents=[
            image,
            "이 스도쿠 판을 분석하여 틀린 숫자와 단 하나의 논리적 힌트를 JSON으로 반환하세요.",
        ],
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=SudokuAnalysis,
        ),
    )

    if not response or not response.text:
        raise RuntimeError("Gemini가 비어 있는 응답을 반환했습니다.")

    return SudokuAnalysis.model_validate_json(response.text)


# -----------------------------------------------------------------------------
# Image helpers
# -----------------------------------------------------------------------------
def normalize_image(image: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(image).convert("RGB")


def resize_image(image: Image.Image, max_dim: int) -> Image.Image:
    image = normalize_image(image)
    width, height = image.size
    if max(width, height) <= max_dim:
        return image
    ratio = max_dim / max(width, height)
    return image.resize(
        (max(1, int(width * ratio)), max(1, int(height * ratio))),
        Image.Resampling.LANCZOS,
    )


def image_digest(image: Image.Image) -> str:
    normalized = normalize_image(image)
    hasher = hashlib.sha256()
    hasher.update(str(normalized.size).encode())
    hasher.update(normalized.tobytes())
    return hasher.hexdigest()


def uploaded_file_digest(uploaded_file) -> str:
    data = uploaded_file.getvalue()
    return hashlib.sha256(data).hexdigest()


def draw_errors_on_image(image: Image.Image, error_cells: list[SudokuError]) -> Image.Image:
    annotated = normalize_image(image).copy()
    draw = ImageDraw.Draw(annotated)
    width, height = annotated.size
    cell_width = width / 9
    cell_height = height / 9
    stroke = max(3, int(width / 80))

    for error in error_cells:
        row, col = error.row - 1, error.col - 1
        x1 = col * cell_width + cell_width * 0.15
        y1 = row * cell_height + cell_height * 0.15
        x2 = (col + 1) * cell_width - cell_width * 0.15
        y2 = (row + 1) * cell_height - cell_height * 0.15
        draw.line([(x1, y1), (x2, y2)], fill="red", width=stroke)
        draw.line([(x1, y2), (x2, y1)], fill="red", width=stroke)

    return annotated


# -----------------------------------------------------------------------------
# Sudoku algorithms
# -----------------------------------------------------------------------------
def is_valid(board: list[list[int]], row: int, col: int, num: int) -> bool:
    for index in range(9):
        if board[row][index] == num or board[index][col] == num:
            return False
        box_row = 3 * (row // 3) + index // 3
        box_col = 3 * (col // 3) + index % 3
        if board[box_row][box_col] == num:
            return False
    return True


def find_empty(board: list[list[int]]):
    best_cell = None
    best_candidates = None
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                candidates = [n for n in range(1, 10) if is_valid(board, row, col, n)]
                if not candidates:
                    return (row, col, [])
                if best_candidates is None or len(candidates) < len(best_candidates):
                    best_cell = (row, col)
                    best_candidates = candidates
                    if len(candidates) == 1:
                        return (row, col, candidates)
    if best_cell is None:
        return None
    return (*best_cell, best_candidates)


def fill_board(board: list[list[int]]) -> bool:
    empty = find_empty(board)
    if empty is None:
        return True
    row, col, candidates = empty
    random.shuffle(candidates)
    for num in candidates:
        board[row][col] = num
        if fill_board(board):
            return True
        board[row][col] = 0
    return False


def count_solutions(board: list[list[int]], limit: int = 2) -> int:
    empty = find_empty(board)
    if empty is None:
        return 1

    row, col, candidates = empty
    total = 0
    for num in candidates:
        board[row][col] = num
        total += count_solutions(board, limit)
        board[row][col] = 0
        if total >= limit:
            return total
    return total


def solve_sudoku_exact(board: list[list[int]]) -> list[list[int]] | None:
    copied = [row[:] for row in board]
    if fill_board(copied):
        return copied
    return None


def generate_sudoku_puzzle(difficulty: str):
    clues_by_difficulty = {"초급": 38, "중급": 30, "고급": 24}
    desired_clues = clues_by_difficulty[difficulty]

    full_board = [[0] * 9 for _ in range(9)]
    fill_board(full_board)
    puzzle = [row[:] for row in full_board]

    positions = [(row, col) for row in range(9) for col in range(9)]
    random.shuffle(positions)
    remaining = 81

    for row, col in positions:
        if remaining <= desired_clues:
            break
        previous = puzzle[row][col]
        puzzle[row][col] = 0
        if count_solutions([r[:] for r in puzzle], limit=2) != 1:
            puzzle[row][col] = previous
        else:
            remaining -= 1

    return puzzle, full_board


# -----------------------------------------------------------------------------
# Local puzzle archive (app server storage only)
# -----------------------------------------------------------------------------
def load_puzzles(difficulty: str | None = None) -> list[dict]:
    if not PUZZLE_FILE.exists():
        return []
    try:
        with PUZZLE_FILE.open("r", encoding="utf-8") as file:
            puzzles = json.load(file)
        if not isinstance(puzzles, list):
            return []
        if difficulty:
            return [item for item in puzzles if item.get("difficulty") == difficulty]
        return puzzles
    except (OSError, json.JSONDecodeError):
        return []


def save_puzzle(difficulty: str, puzzle: list[list[int]], solution: list[list[int]]):
    puzzles = load_puzzles()
    next_id = max((item.get("id", 0) for item in puzzles), default=0) + 1
    puzzles.append(
        {
            "id": next_id,
            "difficulty": difficulty,
            "puzzle": puzzle,
            "solution": solution,
            "created_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )

    temporary_file = PUZZLE_FILE.with_suffix(".tmp")
    with temporary_file.open("w", encoding="utf-8") as file:
        json.dump(puzzles, file, ensure_ascii=False, indent=2)
    temporary_file.replace(PUZZLE_FILE)


# -----------------------------------------------------------------------------
# Board rendering: on-screen HTML, PNG download, A4 PDF download
# -----------------------------------------------------------------------------
def render_sudoku_board_html(puzzle, solution=None):
    html = """
    <style>
        .sudoku-container { display:flex; justify-content:center; margin:15px 0; overflow-x:auto; }
        .sudoku-board { border-collapse:collapse; border:3px solid #222; background:#fff; }
        .sudoku-board td { width:36px; height:36px; text-align:center; vertical-align:middle; border:1px solid #ccc; font-size:18px; font-weight:700; color:#111; }
        .sudoku-board td.solution-cell { color:#1d4ed8; background:#eff6ff; }
        .sudoku-board tr:nth-child(3n) td { border-bottom:2px solid #222; }
        .sudoku-board td:nth-child(3n) { border-right:2px solid #222; }
        .sudoku-board tr:first-child td { border-top:2px solid #222; }
        .sudoku-board td:first-child { border-left:2px solid #222; }
    </style>
    <div class='sudoku-container'><table class='sudoku-board'>
    """
    for row in range(9):
        html += "<tr>"
        for col in range(9):
            value = puzzle[row][col]
            if value:
                html += f"<td>{value}</td>"
            elif solution:
                html += f"<td class='solution-cell'>{solution[row][col]}</td>"
            else:
                html += "<td></td>"
        html += "</tr>"
    return html + "</table></div>"


def load_pil_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def make_board_png(puzzle, solution=None, title="Daily Sudoku Puzzle") -> bytes:
    cell = 72
    margin = 42
    title_height = 72
    board_size = cell * 9
    image = Image.new("RGB", (board_size + margin * 2, board_size + margin * 2 + title_height), "white")
    draw = ImageDraw.Draw(image)
    number_font = load_pil_font(38, bold=True)
    title_font = load_pil_font(30, bold=True)
    draw.text((margin, 18), title, fill="#111827", font=title_font)

    x0, y0 = margin, margin + title_height
    for i in range(10):
        width = 5 if i % 3 == 0 else 1
        draw.line((x0 + i * cell, y0, x0 + i * cell, y0 + board_size), fill="#111111", width=width)
        draw.line((x0, y0 + i * cell, x0 + board_size, y0 + i * cell), fill="#111111", width=width)

    for row in range(9):
        for col in range(9):
            given = puzzle[row][col]
            value = given or (solution[row][col] if solution else 0)
            if not value:
                continue
            color = "#111111" if given else "#1d4ed8"
            bbox = draw.textbbox((0, 0), str(value), font=number_font)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x = x0 + col * cell + (cell - text_w) / 2
            y = y0 + row * cell + (cell - text_h) / 2 - 4
            draw.text((x, y), str(value), fill=color, font=number_font)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def make_print_pdf(puzzle: list[list[int]], print_date: dt.date) -> bytes:
    """Create a print-ready, one-page A4 puzzle PDF. No Drive upload occurs."""
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    page_width, page_height = A4

    title = "Daily Sudoku Puzzle"
    date_text = print_date.strftime("%Y.%m.%d")
    pdf.setFillColor(HexColor("#111827"))
    pdf.setFont("Helvetica-Bold", 26)
    pdf.drawCentredString(page_width / 2, page_height - 30 * mm, title)

    pdf.setFillColor(HexColor("#4B5563"))
    pdf.setFont("Helvetica", 12)
    pdf.drawCentredString(page_width / 2, page_height - 39 * mm, date_text)

    board_size = 160 * mm
    cell = board_size / 9
    board_x = (page_width - board_size) / 2
    board_y = 57 * mm

    pdf.setStrokeColor(HexColor("#111111"))
    for i in range(10):
        pdf.setLineWidth(2.1 if i % 3 == 0 else 0.45)
        position = i * cell
        pdf.line(board_x + position, board_y, board_x + position, board_y + board_size)
        pdf.line(board_x, board_y + position, board_x + board_size, board_y + position)

    pdf.setFillColor(HexColor("#111111"))
    pdf.setFont("Helvetica-Bold", 19)
    for row in range(9):
        for col in range(9):
            value = puzzle[row][col]
            if not value:
                continue
            text = str(value)
            text_width = stringWidth(text, "Helvetica-Bold", 19)
            x = board_x + col * cell + (cell - text_width) / 2
            y = board_y + (8 - row) * cell + cell * 0.31
            pdf.drawString(x, y, text)

    pdf.setFillColor(HexColor("#6B7280"))
    pdf.setFont("Helvetica", 9)
    pdf.drawCentredString(page_width / 2, 24 * mm, "Solve one square at a time. Enjoy your puzzle!")
    pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def device_download_buttons(puzzle, difficulty: str, key_prefix: str):
    st.caption("다운로드 버튼을 누르면 파일이 현재 사용 중인 기기(휴대폰 또는 PC)에 저장됩니다.")
    print_date = st.date_input("인쇄 날짜", value=dt.date.today(), key=f"{key_prefix}_print_date")
    pdf_data = make_print_pdf(puzzle, print_date)
    png_data = make_board_png(puzzle, title=f"Daily Sudoku Puzzle · {difficulty}")
    date_part = print_date.strftime("%Y%m%d")

    left, right = st.columns(2)
    with left:
        st.download_button(
            "🖨️ A4 인쇄용 PDF 저장",
            data=pdf_data,
            file_name=f"daily_sudoku_{date_part}.pdf",
            mime="application/pdf",
            key=f"{key_prefix}_pdf_download",
            use_container_width=True,
        )
    with right:
        st.download_button(
            "🖼️ 문제 PNG 저장",
            data=png_data,
            file_name=f"daily_sudoku_{date_part}.png",
            mime="image/png",
            key=f"{key_prefix}_png_download",
            use_container_width=True,
        )


# -----------------------------------------------------------------------------
# Tab 1: photo analysis
# -----------------------------------------------------------------------------
tab_photo, tab_puzzle = st.tabs(["📸 이미지 업로드 & 도움받기", "🎲 문제 만들기 & 보관함"])

with tab_photo:
    st.subheader("1. 스도쿠 이미지 가져오기")
    uploaded = st.file_uploader("스도쿠 이미지를 촬영하거나 업로드하세요", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        current_upload_hash = uploaded_file_digest(uploaded)
        if st.session_state.get("upload_hash") != current_upload_hash:
            st.session_state["upload_hash"] = current_upload_hash
            st.session_state["rotate_angle"] = 0
            st.session_state.pop("analysis", None)
            st.session_state.pop("analysis_image", None)
            st.session_state.pop("crop_hash", None)

        try:
            raw_image = normalize_image(Image.open(uploaded))
        except (UnidentifiedImageError, OSError):
            st.error("이미지 파일을 열 수 없습니다. JPG 또는 PNG 파일인지 확인해 주세요.")
            st.stop()

        display_image = resize_image(raw_image, DISPLAY_MAX_DIM)
        st.session_state.setdefault("rotate_angle", 0)

        st.subheader("2. 사진 방향 및 영역 설정")
        rotate_col, reset_col = st.columns(2)
        with rotate_col:
            if st.button("🔄 90° 회전", key="rotate_button"):
                st.session_state["rotate_angle"] = (st.session_state["rotate_angle"] - 90) % 360
                st.session_state.pop("analysis", None)
                st.session_state.pop("analysis_image", None)
        with reset_col:
            if st.button("↩️ 방향 초기화", key="reset_rotation_button"):
                st.session_state["rotate_angle"] = 0
                st.session_state.pop("analysis", None)
                st.session_state.pop("analysis_image", None)

        rotation = st.session_state["rotate_angle"]
        if rotation:
            display_image = display_image.rotate(rotation, expand=True)
            api_base_image = raw_image.rotate(rotation, expand=True)
        else:
            api_base_image = raw_image

        use_cropper = st.checkbox("✂️ 빨간 박스로 9x9 영역 잘라내기", value=True, key="use_cropper")
        if use_cropper:
            st.write("모서리를 움직여 스도쿠 9x9 영역에 맞추세요.")
            selected_display_image = st_cropper(
                display_image,
                realtime_update=True,
                box_color="#FF0000",
                aspect_ratio=(1, 1),
                key="cropper_widget",
            )
        else:
            selected_display_image = display_image

        if selected_display_image is not None:
            new_crop_hash = image_digest(selected_display_image)
            if st.session_state.get("crop_hash") != new_crop_hash:
                st.session_state["crop_hash"] = new_crop_hash
                st.session_state.pop("analysis", None)
                st.session_state.pop("analysis_image", None)

            st.image(selected_display_image, caption="최종 분석 영역", use_container_width=True)
            st.caption("정확한 판독을 위해 사진이 흐리지 않고 격자가 정면을 향하도록 촬영해 주세요.")

            api_key = get_api_key()
            if not api_key:
                st.info("Gemini 분석을 사용하려면 Streamlit Secrets에 GEMINI_API_KEY를 설정해 주세요.")
            elif st.button("💡 도움받기: 한 칸 힌트 & 검증", type="primary", key="analyze_button"):
                api_image = resize_image(selected_display_image, API_MAX_DIM)
                with st.spinner("Gemini AI가 스도쿠를 분석하고 있습니다..."):
                    try:
                        client = get_gemini_client(api_key)
                        result = analyze_sudoku(client, api_image, DEFAULT_MODEL)
                        st.session_state["analysis"] = result.model_dump()
                        st.session_state["analysis_image"] = selected_display_image.copy()
                    except Exception as error:
                        st.error(
                            "AI 분석에 실패했습니다. GEMINI_MODEL 설정값, API 키, 모델 사용 권한을 확인한 뒤 다시 시도해 주세요."
                        )
                        st.exception(error)

            if "analysis" in st.session_state and "analysis_image" in st.session_state:
                result = SudokuAnalysis.model_validate(st.session_state["analysis"])
                analyzed_image = st.session_state["analysis_image"]
                st.markdown("---")

                if result.errors:
                    st.error(f"검증 결과: 규칙에 위배되는 숫자가 {len(result.errors)}곳 있습니다.")
                    st.image(
                        draw_errors_on_image(analyzed_image, result.errors),
                        caption="틀린 위치를 빨간 X로 표시했습니다.",
                        use_container_width=True,
                    )
                    for error in result.errors:
                        st.write(f"- {error.row}행 {error.col}열: {error.reason}")
                else:
                    st.success("검증 결과: 현재 적힌 숫자에서 스도쿠 규칙 위반을 찾지 못했습니다.")

                if result.single_hint:
                    hint = result.single_hint
                    st.subheader("💡 바로 해결 가능한 한 칸")
                    st.info(
                        f"위치: {hint.row}행 {hint.col}열\n\n"
                        f"들어갈 숫자: {hint.number}\n\n"
                        f"이유: {hint.reason}"
                    )
                else:
                    st.info("현재 이미지에서는 확실한 한 칸 힌트를 찾지 못했습니다.")


# -----------------------------------------------------------------------------
# Tab 2: puzzle generator, archive, downloads
# -----------------------------------------------------------------------------
with tab_puzzle:
    st.subheader("🎲 난이도별 스도쿠 문제 생성")
    select_col, button_col = st.columns([2, 1])
    with select_col:
        selected_difficulty = st.selectbox("난이도", ["초급", "중급", "고급"])
    with button_col:
        st.write("")
        st.write("")
        generate_clicked = st.button("문제 생성", type="primary", use_container_width=True)

    if generate_clicked:
        with st.spinner("유일한 정답을 가진 문제를 만들고 있습니다..."):
            puzzle, solution = generate_sudoku_puzzle(selected_difficulty)
            try:
                save_puzzle(selected_difficulty, puzzle, solution)
                saved_message = "보관함에도 저장했습니다."
            except OSError:
                saved_message = "문제는 생성했지만 보관함 파일에는 저장하지 못했습니다."
            st.session_state["current_puzzle"] = puzzle
            st.session_state["current_solution"] = solution
            st.session_state["current_difficulty"] = selected_difficulty
            st.success(f"새로운 {selected_difficulty} 문제를 생성했습니다. {saved_message}")

    if "current_puzzle" in st.session_state:
        current_puzzle = st.session_state["current_puzzle"]
        current_solution = st.session_state["current_solution"]
        current_difficulty = st.session_state["current_difficulty"]
        st.markdown(f"### 📋 생성된 문제 · {current_difficulty}")
        show_solution = st.toggle("🔍 정답 보기", key="current_solution_toggle")
        st.markdown(
            render_sudoku_board_html(current_puzzle, current_solution if show_solution else None),
            unsafe_allow_html=True,
        )
        device_download_buttons(current_puzzle, current_difficulty, "current")

    st.markdown("---")
    st.subheader("📁 저장된 문제 보관함")
    archive_filter = st.radio("조회 난이도", ["전체", "초급", "중급", "고급"], horizontal=True)
    filtered_difficulty = None if archive_filter == "전체" else archive_filter
    archived_puzzles = load_puzzles(filtered_difficulty)

    if not archived_puzzles:
        st.info("저장된 문제가 없습니다. 위에서 새 문제를 만들어 보세요.")
    else:
        st.write(f"총 {len(archived_puzzles)}개의 저장된 문제가 있습니다.")
        selected_index = st.selectbox(
            "불러올 문제",
            options=range(len(archived_puzzles)),
            format_func=lambda index: (
                f"#{archived_puzzles[index].get('id', '?')} "
                f"[{archived_puzzles[index].get('difficulty', '')}] "
                f"{archived_puzzles[index].get('created_at', '')}"
            ),
        )
        saved_item = archived_puzzles[selected_index]
        saved_puzzle = saved_item["puzzle"]
        saved_solution = saved_item.get("solution") or solve_sudoku_exact(saved_puzzle)
        saved_difficulty = saved_item.get("difficulty", "스도쿠")
        show_saved_solution = st.toggle("🔍 저장된 문제 정답 보기", key="saved_solution_toggle")
        st.markdown(
            render_sudoku_board_html(saved_puzzle, saved_solution if show_saved_solution else None),
            unsafe_allow_html=True,
        )
        device_download_buttons(saved_puzzle, saved_difficulty, f"archive_{saved_item.get('id', selected_index)}")
