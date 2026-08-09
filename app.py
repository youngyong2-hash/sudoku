import os
import io
import json
import random
import hashlib
import datetime
from typing import Literal, Optional

import streamlit as st
from PIL import Image, ImageDraw, ImageOps, ImageFont
from streamlit_cropper import st_cropper
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
st.set_page_config(page_title="스도쿠 AI 도우미", page_icon="🧩", layout="centered")

st.markdown(
    """
    <style>
        .stApp { max-width: 100%; padding-left: 0.5rem; padding-right: 0.5rem; }
        iframe { max-width: 100% !important; width: 100% !important; }
        img { max-width: 100% !important; height: auto !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

PUZZLE_FILE = "puzzles_db.json"

# 2026년 8월 기준 Gemini 2.x Flash 계열은 신규 사용자에게 404(NOT_FOUND)를 반환할 수 있으므로
# 사용하지 않는다. 아래 3.x Flash 계열만 순서대로 시도한다.
MODEL_CANDIDATES = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
]


# ------------------------------------------------------------------------------
# 2. Gemini response schema
# ------------------------------------------------------------------------------
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
    mode: Literal["check_answer", "help"]
    errors: list[SudokuError] = Field(default_factory=list)
    single_hint: Optional[SudokuHint] = None
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
# 3. Image and storage helpers
# ------------------------------------------------------------------------------
def image_hash(image: Image.Image) -> str:
    normalized = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(normalized.size.__repr__().encode("utf-8"))
    digest.update(normalized.tobytes())
    return digest.hexdigest()


def uploaded_file_hash(uploaded_file) -> str:
    data = uploaded_file.getvalue()
    return hashlib.sha256(data).hexdigest()


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


def load_puzzles(difficulty=None):
    if not os.path.exists(PUZZLE_FILE):
        return []
    try:
        with open(PUZZLE_FILE, "r", encoding="utf-8") as file:
            puzzles = json.load(file)
        if not isinstance(puzzles, list):
            return []
        if difficulty:
            return [item for item in puzzles if item.get("difficulty") == difficulty]
        return puzzles
    except (OSError, json.JSONDecodeError, ValueError):
        return []


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

    temp_file = f"{PUZZLE_FILE}.tmp"
    try:
        with open(temp_file, "w", encoding="utf-8") as file:
            json.dump(puzzles, file, ensure_ascii=False, indent=2)
        os.replace(temp_file, PUZZLE_FILE)
    except OSError as error:
        st.error(f"데이터 저장 중 오류가 발생했습니다: {error}")
        if os.path.exists(temp_file):
            os.remove(temp_file)


# ------------------------------------------------------------------------------
# 4. Sudoku algorithms
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
    """A4 용지에 제목(Daily Sudoku Puzzle), 날짜, 가운데 정렬된 스도쿠 판을 그려 인쇄용 PDF 바이트를 반환한다."""
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


def draw_errors_on_image(image, error_cells):
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    width, height = annotated.size
    cell_w, cell_h = width / 9.0, height / 9.0

    for item in error_cells:
        row, col = item.get("row", 0), item.get("col", 0)
        if 1 <= row <= 9 and 1 <= col <= 9:
            row_idx, col_idx = row - 1, col - 1
            x1 = col_idx * cell_w + cell_w * 0.15
            y1 = row_idx * cell_h + cell_h * 0.15
            x2 = (col_idx + 1) * cell_w - cell_w * 0.15
            y2 = (row_idx + 1) * cell_h - cell_h * 0.15
            stroke = max(3, int(width / 80))
            draw.line([(x1, y1), (x2, y2)], fill="red", width=stroke)
            draw.line([(x1, y2), (x2, y1)], fill="red", width=stroke)
    return annotated


# ------------------------------------------------------------------------------
# 6. Gemini analysis (automatic model fallback avoids 404 errors)
# ------------------------------------------------------------------------------
SYSTEM_PROMPT = """
당신은 엄격하고 명확한 스도쿠 검증 튜터입니다.
업로드된 이미지에서 9x9 스도쿠 판(인쇄체 및 손글씨)을 분석합니다.

먼저 9x9 모든 칸이 숫자로 채워졌는지 판단하세요. 숫자가 불분명한 칸은 빈칸으로 간주합니다.

분기 규칙:
1. 모든 칸이 채워진 완성판이면 mode를 check_answer로 설정합니다.
   - 스도쿠 전체 정답을 검증합니다.
   - 틀린 숫자의 위치를 errors에 모두 기록합니다.
   - 정답이면 errors는 빈 배열입니다.
   - single_hint는 null입니다.
2. 하나라도 빈칸이 있으면 mode를 help로 설정합니다.
   - 현재 적힌 숫자 중 스도쿠 규칙에 위배되는 숫자를 errors에 기록합니다.
   - 빈칸 중 논리적으로 확실히 채울 수 있는 칸이 있으면 single_hint에 단 한 칸만 제시합니다.
   - 확실한 한 칸 힌트가 없으면 single_hint는 null입니다.

행과 열은 항상 1~9 정수로 반환합니다. reason과 message는 한국어로 간결하게 작성합니다.
"""


def analyze_sudoku(client, image):
    """MODEL_CANDIDATES를 순서대로 시도해 404(모델 미제공) 오류에도 자동으로 복구한다."""
    last_error = None

    for model_name in MODEL_CANDIDATES:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    image,
                    "이미지의 스도쿠를 판독하고, 완성 여부에 따라 정답 확인 또는 풀이 도움 결과를 반환하세요.",
                ],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema=SudokuAnalysis,
                    max_output_tokens=500,
                ),
            )
            if not response or not response.text:
                raise ValueError("Gemini가 빈 응답을 반환했습니다.")
            parsed = SudokuAnalysis.model_validate_json(response.text).model_dump()
            return parsed, model_name
        except genai_errors.ClientError as error:
            last_error = error
            if getattr(error, "code", None) == 404 or "NOT_FOUND" in str(error):
                continue
            raise
        except Exception as error:
            last_error = error
            continue

    raise RuntimeError(f"사용 가능한 Gemini 모델을 찾지 못했습니다: {last_error}")


def clear_analysis():
    st.session_state.pop("ai_analysis_result", None)
    st.session_state.pop("cropped_img_for_display", None)
    st.session_state.pop("ai_used_model", None)


# ------------------------------------------------------------------------------
# 7. Main UI
# ------------------------------------------------------------------------------
st.title("🧩 스도쿠 AI 스마트 도우미")

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
            st.session_state["rotate_angle"] = 0
            st.session_state.pop("last_crop_hash", None)
            clear_analysis()

        try:
            raw_image = Image.open(io.BytesIO(img_file.getvalue()))
            working_img = resize_image(raw_image, max_dim=768)
        except (OSError, ValueError) as error:
            st.error(f"이미지를 열 수 없습니다. JPG 또는 PNG 파일을 확인해 주세요: {error}")
            st.stop()

        st.session_state.setdefault("rotate_angle", 0)
        st.subheader("2. 사진 방향 및 영역 설정")
        rotate_col, reset_col = st.columns(2)
        with rotate_col:
            if st.button("🔄 90° 회전"):
                st.session_state["rotate_angle"] = (st.session_state["rotate_angle"] - 90) % 360
                clear_analysis()
        with reset_col:
            if st.button("↩️ 방향 초기화"):
                st.session_state["rotate_angle"] = 0
                clear_analysis()

        if st.session_state["rotate_angle"]:
            working_img = working_img.rotate(st.session_state["rotate_angle"], expand=True)

        use_cropper = st.checkbox("✂️ 빨간 박스로 9×9 영역 잘라내기", value=True)
        if st.session_state.get("last_use_cropper") != use_cropper:
            st.session_state["last_use_cropper"] = use_cropper
            clear_analysis()

        target_img = working_img
        if use_cropper:
            st.write("📱 모서리를 조절해 9×9 스도쿠 영역에 맞추세요.")
            target_img = st_cropper(
                working_img,
                realtime_update=True,
                box_color="#FF0000",
                aspect_ratio=(1, 1),
                key="cropper_widget",
            )

        if target_img is not None:
            crop_hash = image_hash(target_img)
            if st.session_state.get("last_crop_hash") != crop_hash:
                st.session_state["last_crop_hash"] = crop_hash
                clear_analysis()

            st.image(target_img, caption="최종 분석 영역", use_container_width=True)

            if st.button("💡 판독 후 정답확인 또는 도움받기", type="primary"):
                with st.spinner("Gemini AI가 스도쿠를 분석하고 있습니다..."):
                    try:
                        result, used_model = analyze_sudoku(client, target_img)
                        st.session_state["ai_analysis_result"] = result
                        st.session_state["cropped_img_for_display"] = target_img.copy()
                        st.session_state["ai_used_model"] = used_model
                    except Exception as error:
                        st.error(f"AI 분석 중 오류가 발생했습니다: {error}")

            if "ai_analysis_result" in st.session_state and "cropped_img_for_display" in st.session_state:
                result = st.session_state["ai_analysis_result"]
                saved_img = st.session_state["cropped_img_for_display"]
                errors = result.get("errors", [])
                hint = result.get("single_hint")
                mode = result.get("mode", "help")
                message = result.get("message", "")
                st.markdown("---")

                if mode == "check_answer":
                    st.subheader("📝 정답 확인 결과")
                    if errors:
                        st.error(f"❌ 정답이 아닙니다. 틀린 숫자가 {len(errors)}곳 있습니다.")
                        st.image(
                            draw_errors_on_image(saved_img, errors),
                            caption="❌ 틀린 위치가 빨간색 X로 표시되었습니다",
                            use_container_width=True,
                        )
                        for error in errors:
                            st.write(f"- **{error.get('row')}행 {error.get('col')}열**: {error.get('reason')}")
                    else:
                        st.balloons()
                        st.success("🎉 축하합니다! 스도쿠를 정확히 완성했습니다.")
                    if message:
                        st.caption(message)

                else:
                    st.subheader("💡 풀이 도움 결과")
                    if errors:
                        st.error(f"⚠️ 현재 입력 중 규칙에 맞지 않는 숫자가 {len(errors)}곳 있습니다.")
                        st.image(
                            draw_errors_on_image(saved_img, errors),
                            caption="❌ 규칙에 맞지 않는 위치가 빨간색 X로 표시되었습니다",
                            use_container_width=True,
                        )
                        for error in errors:
                            st.write(f"- **{error.get('row')}행 {error.get('col')}열**: {error.get('reason')}")
                    else:
                        st.success("✅ 현재 입력된 숫자에는 규칙상 문제가 없습니다.")

                    if hint:
                        st.subheader("💡 바로 해결 가능한 한 칸 힌트")
                        st.info(
                            f"👉 **위치:** {hint.get('row')}행 {hint.get('col')}열\n\n"
                            f"👉 **넣을 숫자:** {hint.get('number')}\n\n"
                            f"👉 **풀이 이유:** {hint.get('reason')}"
                        )
                    else:
                        st.warning("현재 상태에서는 바로 확정할 수 있는 한 칸을 찾지 못했습니다.")
                    if message:
                        st.caption(message)

                used_model = st.session_state.get("ai_used_model")
                if used_model:
                    st.caption(f"사용된 모델: {used_model}")

# ==============================================================================
# TAB 2
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

    # 탭을 처음 열었을 때도 인쇄용 PDF/PNG 섹션이 바로 보이도록 기본 문제를 자동 생성한다.
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
    else:
        st.info("선택한 난이도에 저장된 스도쿠 문제가 없습니다. 위에서 새 문제를 만들어 보세요!")
