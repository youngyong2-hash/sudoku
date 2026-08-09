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

# =============================================================================
# 1. App settings
# =============================================================================
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
CROPPER_MAX_DIM = 768
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")


# =============================================================================
# 2. Gemini data schema
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
    # 0 means an empty / unreadable cell.
    grid: list[list[int]]
    errors: list[SudokuError] = Field(default_factory=list)
    single_hint: SudokuHint | None = None


def validate_sudoku_grid(grid: list[list[int]]) -> list[list[int]]:
    if len(grid) != 9 or any(len(row) != 9 for row in grid):
        raise ValueError("AI가 9x9 형식이 아닌 데이터를 반환했습니다.")
    for row in grid:
        for number in row:
            if not isinstance(number, int) or not 0 <= number <= 9:
                raise ValueError("스도쿠 숫자는 0부터 9까지의 정수여야 합니다.")
    return grid


@st.cache_resource
def get_gemini_client(api_key: str):
    return genai.Client(api_key=api_key)


def get_api_key() -> str | None:
    return st.secrets.get("GEMINI_API_KEY", None) or os.getenv("GEMINI_API_KEY")


def analyze_sudoku(client, image: Image.Image, model_name: str) -> SudokuAnalysis:
    system_prompt = """
당신은 스도쿠 이미지 판독 및 검증 전문 AI입니다.
이미지에는 하나의 9x9 스도쿠 판이 있으며, 인쇄 숫자와 손글씨 숫자가 섞여 있을 수 있습니다.

가장 중요한 작업은 손글씨와 인쇄 숫자를 읽어 정확한 9x9 grid로 변환하는 것입니다.

반드시 지킬 규칙:
1. grid는 정확히 9개의 행입니다.
2. 각 행은 왼쪽에서 오른쪽 순서의 숫자 9개입니다.
3. 비어 있는 칸 또는 읽기 불확실한 칸은 반드시 0으로 기록합니다.
4. 판독되는 인쇄 숫자와 손글씨 숫자는 1~9 정수로 기록합니다.
5. 확실하지 않은 숫자를 추측하지 마세요. 0으로 남기세요.
6. errors에는 현재 숫자 중 행, 열 또는 3x3 박스 규칙을 위배하는 숫자만 넣습니다.
7. single_hint에는 현재 grid에서 논리적으로 확실히 채울 수 있는 단 한 칸만 넣습니다.
8. 확실한 힌트가 없으면 single_hint를 null로 설정합니다.
9. row와 col은 사람 기준으로 1부터 9까지입니다.
10. 격자 바깥의 제목, 날짜, 낙서, 메모는 무시하세요.
"""

    response = client.models.generate_content(
        model=model_name,
        contents=[image, "이 잘라낸 스도쿠 사진을 읽어 9x9 grid와 검증 결과를 JSON으로 반환하세요."],
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=SudokuAnalysis,
        ),
    )

    if not response or not response.text:
        raise RuntimeError("Gemini가 비어 있는 응답을 반환했습니다.")

    result = SudokuAnalysis.model_validate_json(response.text)
    result.grid = validate_sudoku_grid(result.grid)
    return result


# =============================================================================
# 3. Image helpers
# =============================================================================
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


def image_hash(image: Image.Image) -> str:
    image = normalize_image(image)
    digest = hashlib.sha256()
    digest.update(str(image.size).encode("utf-8"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def upload_hash(uploaded_file) -> str:
    return hashlib.sha256(uploaded_file.getvalue()).hexdigest()


def draw_errors_on_image(image: Image.Image, errors: list[SudokuError]) -> Image.Image:
    result = normalize_image(image).copy()
    draw = ImageDraw.Draw(result)
    width, height = result.size
    cell_width, cell_height = width / 9, height / 9
    line_width = max(3, int(width / 80))

    for error in errors:
        row, col = error.row - 1, error.col - 1
        x1 = col * cell_width + cell_width * 0.15
        y1 = row * cell_height + cell_height * 0.15
        x2 = (col + 1) * cell_width - cell_width * 0.15
        y2 = (row + 1) * cell_height - cell_height * 0.15
        draw.line([(x1, y1), (x2, y2)], fill="red", width=line_width)
        draw.line([(x1, y2), (x2, y1)], fill="red", width=line_width)
    return result


# =============================================================================
# 4. Sudoku generation and solving
# =============================================================================
def is_valid(board: list[list[int]], row: int, col: int, number: int) -> bool:
    for index in range(9):
        if board[row][index] == number or board[index][col] == number:
            return False
        box_row = 3 * (row // 3) + index // 3
        box_col = 3 * (col // 3) + index % 3
        if board[box_row][box_col] == number:
            return False
    return True


def find_best_empty(board: list[list[int]]):
    best = None
    best_candidates = None
    for row in range(9):
        for col in range(9):
            if board[row][col] != 0:
                continue
            candidates = [number for number in range(1, 10) if is_valid(board, row, col, number)]
            if not candidates:
                return row, col, []
            if best_candidates is None or len(candidates) < len(best_candidates):
                best = (row, col)
                best_candidates = candidates
                if len(candidates) == 1:
                    return row, col, candidates
    if best is None:
        return None
    return best[0], best[1], best_candidates


def fill_board(board: list[list[int]]) -> bool:
    empty = find_best_empty(board)
    if empty is None:
        return True
    row, col, candidates = empty
    random.shuffle(candidates)
    for number in candidates:
        board[row][col] = number
        if fill_board(board):
            return True
        board[row][col] = 0
    return False


def count_solutions(board: list[list[int]], limit: int = 2) -> int:
    empty = find_best_empty(board)
    if empty is None:
        return 1
    row, col, candidates = empty
    total = 0
    for number in candidates:
        board[row][col] = number
        total += count_solutions(board, limit)
        board[row][col] = 0
        if total >= limit:
            return total
    return total


def solve_sudoku_exact(board: list[list[int]]) -> list[list[int]] | None:
    copied = [row[:] for row in board]
    return copied if fill_board(copied) else None


def generate_sudoku_puzzle(difficulty: str):
    clues_by_difficulty = {"초급": 38, "중급": 30, "고급": 24}
    desired_clues = clues_by_difficulty[difficulty]

    solution = [[0] * 9 for _ in range(9)]
    fill_board(solution)
    puzzle = [row[:] for row in solution]
    positions = [(row, col) for row in range(9) for col in range(9)]
    random.shuffle(positions)
    clues_left = 81

    for row, col in positions:
        if clues_left <= desired_clues:
            break
        old_value = puzzle[row][col]
        puzzle[row][col] = 0
        if count_solutions([r[:] for r in puzzle], limit=2) == 1:
            clues_left -= 1
        else:
            puzzle[row][col] = old_value
    return puzzle, solution


# =============================================================================
# 5. Local archive (no Google Drive upload)
# =============================================================================
def load_puzzles(difficulty: str | None = None) -> list[dict]:
    if not PUZZLE_FILE.exists():
        return []
    try:
        with PUZZLE_FILE.open("r", encoding="utf-8") as file:
            records = json.load(file)
        if not isinstance(records, list):
            return []
        return [record for record in records if record.get("difficulty") == difficulty] if difficulty else records
    except (OSError, json.JSONDecodeError):
        return []


def save_puzzle(difficulty: str, puzzle: list[list[int]], solution: list[list[int]]):
    records = load_puzzles()
    next_id = max((record.get("id", 0) for record in records), default=0) + 1
    records.append(
        {
            "id": next_id,
            "difficulty": difficulty,
            "puzzle": puzzle,
            "solution": solution,
            "created_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    temp_file = PUZZLE_FILE.with_suffix(".tmp")
    with temp_file.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)
    temp_file.replace(PUZZLE_FILE)


# =============================================================================
# 6. Board rendering and device downloads
# =============================================================================
def render_sudoku_board_html(puzzle: list[list[int]], solution: list[list[int]] | None = None) -> str:
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


def get_font(size: int, bold: bool = False):
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for path in paths:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def make_board_png(puzzle: list[list[int]], solution: list[list[int]] | None = None, title: str = "Daily Sudoku Puzzle") -> bytes:
    cell, margin, title_height = 72, 42, 72
    board_size = cell * 9
    image = Image.new("RGB", (board_size + margin * 2, board_size + margin * 2 + title_height), "white")
    draw = ImageDraw.Draw(image)
    title_font, number_font = get_font(28, bold=True), get_font(38, bold=True)
    draw.text((margin, 18), title, fill="#111827", font=title_font)

    x0, y0 = margin, margin + title_height
    for index in range(10):
        line_width = 5 if index % 3 == 0 else 1
        draw.line((x0 + index * cell, y0, x0 + index * cell, y0 + board_size), fill="#111", width=line_width)
        draw.line((x0, y0 + index * cell, x0 + board_size, y0 + index * cell), fill="#111", width=line_width)

    for row in range(9):
        for col in range(9):
            is_given = puzzle[row][col] != 0
            value = puzzle[row][col] or (solution[row][col] if solution else 0)
            if not value:
                continue
            box = draw.textbbox((0, 0), str(value), font=number_font)
            text_width, text_height = box[2] - box[0], box[3] - box[1]
            x = x0 + col * cell + (cell - text_width) / 2
            y = y0 + row * cell + (cell - text_height) / 2 - 4
            draw.text((x, y), str(value), fill="#111" if is_given else "#1d4ed8", font=number_font)

    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()


def make_print_pdf(puzzle: list[list[int]], print_date: dt.date) -> bytes:
    """One-page A4 PDF. This is downloaded to the user's device, not Drive."""
    output = io.BytesIO()
    pdf = canvas.Canvas(output, pagesize=A4)
    page_width, page_height = A4

    pdf.setFillColor(HexColor("#111827"))
    pdf.setFont("Helvetica-Bold", 26)
    pdf.drawCentredString(page_width / 2, page_height - 30 * mm, "Daily Sudoku Puzzle")
    pdf.setFillColor(HexColor("#4B5563"))
    pdf.setFont("Helvetica", 12)
    pdf.drawCentredString(page_width / 2, page_height - 39 * mm, print_date.strftime("%Y.%m.%d"))

    board_size = 160 * mm
    cell = board_size / 9
    left = (page_width - board_size) / 2
    bottom = 57 * mm
    pdf.setStrokeColor(HexColor("#111111"))
    for index in range(10):
        pdf.setLineWidth(2.1 if index % 3 == 0 else 0.45)
        position = index * cell
        pdf.line(left + position, bottom, left + position, bottom + board_size)
        pdf.line(left, bottom + position, left + board_size, bottom + position)

    pdf.setFillColor(HexColor("#111111"))
    pdf.setFont("Helvetica-Bold", 19)
    for row in range(9):
        for col in range(9):
            value = puzzle[row][col]
            if value:
                text = str(value)
                x = left + col * cell + (cell - stringWidth(text, "Helvetica-Bold", 19)) / 2
                y = bottom + (8 - row) * cell + cell * 0.31
                pdf.drawString(x, y, text)

    pdf.setFillColor(HexColor("#6B7280"))
    pdf.setFont("Helvetica", 9)
    pdf.drawCentredString(page_width / 2, 24 * mm, "Solve one square at a time. Enjoy your puzzle!")
    pdf.showPage()
    pdf.save()
    return output.getvalue()


def show_device_downloads(puzzle: list[list[int]], difficulty: str, key_prefix: str):
    st.caption("버튼을 누르면 PDF 또는 PNG 파일이 현재 사용 중인 기기(휴대폰·PC)에 저장됩니다.")
    date_value = st.date_input("인쇄 날짜", value=dt.date.today(), key=f"{key_prefix}_date")
    file_date = date_value.strftime("%Y%m%d")
    pdf_data = make_print_pdf(puzzle, date_value)
    png_data = make_board_png(puzzle, title=f"Daily Sudoku Puzzle · {difficulty}")
    pdf_col, png_col = st.columns(2)
    with pdf_col:
        st.download_button(
            "🖨️ A4 PDF 저장",
            data=pdf_data,
            file_name=f"daily_sudoku_{file_date}.pdf",
            mime="application/pdf",
            key=f"{key_prefix}_pdf",
            use_container_width=True,
        )
    with png_col:
        st.download_button(
            "🖼️ PNG 저장",
            data=png_data,
            file_name=f"daily_sudoku_{file_date}.png",
            mime="image/png",
            key=f"{key_prefix}_png",
            use_container_width=True,
        )


# =============================================================================
# 7. Main user interface
# =============================================================================
tab_image, tab_create = st.tabs(["📸 사진 읽기 & 도움받기", "🎲 문제 만들기 & 보관함"])

with tab_image:
    st.subheader("1. 스도쿠 사진 가져오기")
    uploaded_image = st.file_uploader("스도쿠 사진을 촬영하거나 업로드하세요", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        new_upload_hash = upload_hash(uploaded_image)
        if st.session_state.get("uploaded_image_hash") != new_upload_hash:
            st.session_state["uploaded_image_hash"] = new_upload_hash
            st.session_state["rotate_angle"] = 0
            st.session_state.pop("crop_image_hash", None)
            st.session_state.pop("ai_analysis", None)
            st.session_state.pop("ai_image", None)

        try:
            raw_image = normalize_image(Image.open(uploaded_image))
        except (UnidentifiedImageError, OSError):
            st.error("이미지를 열 수 없습니다. JPG 또는 PNG 파일인지 확인해 주세요.")
            st.stop()

        st.session_state.setdefault("rotate_angle", 0)
        st.subheader("2. 사진 방향 및 9×9 영역 설정")
        rotate_col, reset_col = st.columns(2)
        with rotate_col:
            if st.button("🔄 90° 회전", key="rotate"):
                st.session_state["rotate_angle"] = (st.session_state["rotate_angle"] - 90) % 360
                st.session_state.pop("ai_analysis", None)
                st.session_state.pop("ai_image", None)
        with reset_col:
            if st.button("↩️ 방향 초기화", key="reset"):
                st.session_state["rotate_angle"] = 0
                st.session_state.pop("ai_analysis", None)
                st.session_state.pop("ai_image", None)

        work_image = resize_image(raw_image, CROPPER_MAX_DIM)
        if st.session_state["rotate_angle"]:
            work_image = work_image.rotate(st.session_state["rotate_angle"], expand=True)

        use_cropper = st.checkbox("✂️ 빨간 박스로 9×9 영역 자르기", value=True, key="crop_enabled")
        if use_cropper:
            st.write("빨간 박스가 스도쿠의 바깥 격자선에 맞도록 모서리를 조절하세요.")
            target_image = st_cropper(
                work_image,
                realtime_update=True,
                box_color="#FF0000",
                aspect_ratio=(1, 1),
                key="sudoku_cropper",
            )
        else:
            target_image = work_image

        if target_image is not None:
            new_crop_hash = image_hash(target_image)
            if st.session_state.get("crop_image_hash") != new_crop_hash:
                st.session_state["crop_image_hash"] = new_crop_hash
                st.session_state.pop("ai_analysis", None)
                st.session_state.pop("ai_image", None)

            st.image(target_image, caption="AI가 읽을 최종 9×9 영역", use_container_width=True)
            st.caption("손글씨 숫자는 굵고 선명하게, 격자가 정면을 보도록 촬영하면 인식률이 좋아집니다.")

            api_key = get_api_key()
            model_name = st.text_input("Gemini 모델", value=DEFAULT_MODEL, key="gemini_model")
            if not api_key:
                st.info("Gemini 분석을 사용하려면 Streamlit Secrets에 GEMINI_API_KEY를 설정해 주세요.")
            elif st.button("🔎 손글씨 읽기 · 9×9 변환 · 검증", type="primary", key="read_sudoku"):
                with st.spinner("손글씨 숫자를 읽고 9×9 스도쿠 판으로 변환하고 있습니다..."):
                    try:
                        client = get_gemini_client(api_key)
                        analysis = analyze_sudoku(client, target_image, model_name)
                        st.session_state["ai_analysis"] = analysis.model_dump()
                        st.session_state["ai_image"] = target_image.copy()
                    except Exception as error:
                        st.error("AI 분석에 실패했습니다. API 키, 모델명, 모델 사용 권한을 확인해 주세요.")
                        st.exception(error)

            if "ai_analysis" in st.session_state and "ai_image" in st.session_state:
                analysis = SudokuAnalysis.model_validate(st.session_state["ai_analysis"])
                recognized_grid = validate_sudoku_grid(analysis.grid)
                saved_image = st.session_state["ai_image"]
                st.markdown("---")
                st.subheader("🔎 AI가 읽은 9×9 스도쿠 판")
                st.markdown(render_sudoku_board_html(recognized_grid), unsafe_allow_html=True)
                st.caption("빈칸 또는 판독이 불확실한 칸은 비어 있는 칸(내부값 0)으로 표시됩니다.")

                grid_png = make_board_png(recognized_grid, title="AI Read Sudoku Grid")
                st.download_button(
                    "🖼️ 인식된 9×9 판 PNG 저장",
                    data=grid_png,
                    file_name="recognized_sudoku_grid.png",
                    mime="image/png",
                    key="recognized_grid_png",
                    use_container_width=True,
                )

                if analysis.errors:
                    st.error(f"검증 결과: 규칙에 위배되는 숫자가 {len(analysis.errors)}곳 있습니다.")
                    st.image(
                        draw_errors_on_image(saved_image, analysis.errors),
                        caption="규칙에 맞지 않는 위치를 빨간 X로 표시했습니다.",
                        use_container_width=True,
                    )
                    for error in analysis.errors:
                        st.write(f"- {error.row}행 {error.col}열: {error.reason}")
                else:
                    st.success("검증 결과: 현재 읽힌 숫자에서 스도쿠 규칙 위반을 찾지 못했습니다.")

                if analysis.single_hint:
                    hint = analysis.single_hint
                    st.subheader("💡 바로 해결 가능한 한 칸")
                    st.info(
                        f"위치: {hint.row}행 {hint.col}열\n\n"
                        f"들어갈 숫자: {hint.number}\n\n"
                        f"이유: {hint.reason}"
                    )
                else:
                    st.info("이 사진에서는 확실한 한 칸 힌트를 찾지 못했습니다.")

with tab_create:
    st.subheader("🎲 난이도별 스도쿠 문제 생성")
    difficulty_col, create_col = st.columns([2, 1])
    with difficulty_col:
        difficulty = st.selectbox("난이도", ["초급", "중급", "고급"])
    with create_col:
        st.write("")
        st.write("")
        create_button = st.button("문제 생성", type="primary", use_container_width=True)

    if create_button:
        with st.spinner("유일한 정답을 가진 문제를 만들고 있습니다..."):
            puzzle, solution = generate_sudoku_puzzle(difficulty)
            st.session_state["current_puzzle"] = puzzle
            st.session_state["current_solution"] = solution
            st.session_state["current_difficulty"] = difficulty
            try:
                save_puzzle(difficulty, puzzle, solution)
                st.success(f"새로운 {difficulty} 문제를 만들고 보관함에 저장했습니다.")
            except OSError:
                st.warning("문제는 생성했지만 서버 보관함 파일 저장에는 실패했습니다. 다운로드는 가능합니다.")

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
        show_device_downloads(current_puzzle, current_difficulty, "current")

    st.markdown("---")
    st.subheader("📁 저장된 문제 보관함")
    filter_option = st.radio("조회 난이도", ["전체", "초급", "중급", "고급"], horizontal=True)
    records = load_puzzles(None if filter_option == "전체" else filter_option)

    if not records:
        st.info("저장된 문제가 없습니다. 위에서 새 문제를 만들어 보세요.")
    else:
        st.write(f"총 {len(records)}개의 문제가 저장되어 있습니다.")
        selected_index = st.selectbox(
            "불러올 문제",
            options=range(len(records)),
            format_func=lambda index: (
                f"#{records[index].get('id', '?')} "
                f"[{records[index].get('difficulty', '')}] "
                f"{records[index].get('created_at', '')}"
            ),
        )
        record = records[selected_index]
        saved_puzzle = record["puzzle"]
        saved_solution = record.get("solution") or solve_sudoku_exact(saved_puzzle)
        saved_difficulty = record.get("difficulty", "스도쿠")
        show_saved_solution = st.toggle("🔍 저장된 문제 정답 보기", key="saved_solution_toggle")
        st.markdown(
            render_sudoku_board_html(saved_puzzle, saved_solution if show_saved_solution else None),
            unsafe_allow_html=True,
        )
        show_device_downloads(saved_puzzle, saved_difficulty, f"saved_{record.get('id', selected_index)}")
