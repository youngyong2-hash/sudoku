import os, io, json, random, hashlib
import datetime as dt
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps
from streamlit_cropper import st_cropper
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas
from reportlab.pdfbase.pdfmetrics import stringWidth

# =============================================================================
# 설정
# =============================================================================
st.set_page_config(page_title="영용's Sudoku", page_icon="🏄", layout="centered")
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

DB_FILE = Path("puzzles_db.json")
MAX_IMAGE_DIM = 768
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

# =============================================================================
# Gemini: 손글씨 판독 JSON 형식
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
def gemini_client(api_key):
    return genai.Client(api_key=api_key)

def get_api_key():
    return st.secrets.get("GEMINI_API_KEY", None) or os.getenv("GEMINI_API_KEY")

def validate_grid(grid):
    if len(grid) != 9 or any(len(row) != 9 for row in grid):
        raise ValueError("AI가 9x9 형식이 아닌 데이터를 반환했습니다.")
    if any(not isinstance(n, int) or n < 0 or n > 9 for row in grid for n in row):
        raise ValueError("스도쿠 숫자는 0부터 9까지만 허용됩니다.")
    return grid

def read_sudoku_with_ai(client, image, model):
    prompt = """
당신은 스도쿠 사진 판독 전문 AI입니다. 이미지의 9x9 스도쿠 판을 읽으세요.
인쇄 숫자와 손글씨 숫자를 모두 읽어 grid에 기록합니다.
- grid는 정확히 9행 9열이며, 왼쪽에서 오른쪽, 위에서 아래 순서입니다.
- 비어 있거나 숫자를 확신할 수 없는 칸은 0입니다. 추측하지 마세요.
- errors와 single_hint도 JSON 스키마에 맞춰 반환하세요.
- 격자 밖의 글, 메모, 날짜는 무시하세요.
"""
    response = client.models.generate_content(
        model=model,
        contents=[image, "사진 속 스도쿠를 9x9 숫자 배열로 읽어 JSON으로 반환하세요."],
        config=types.GenerateContentConfig(
            system_instruction=prompt,
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=SudokuAnalysis,
        ),
    )
    if not response or not response.text:
        raise RuntimeError("Gemini가 빈 응답을 반환했습니다.")
    data = SudokuAnalysis.model_validate_json(response.text)
    data.grid = validate_grid(data.grid)
    return data

# =============================================================================
# 이미지 유틸
# =============================================================================
def normalize(image):
    return ImageOps.exif_transpose(image).convert("RGB")

def resize(image, max_dim=MAX_IMAGE_DIM):
    image = normalize(image)
    w, h = image.size
    if max(w, h) <= max_dim:
        return image
    ratio = max_dim / max(w, h)
    return image.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)

def digest_image(image):
    image = normalize(image)
    return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()

def mark_photo_errors(image, errors):
    out = normalize(image).copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    cw, ch = w / 9, h / 9
    stroke = max(3, int(w / 80))
    for row, col in errors:
        x1, y1 = col*cw+cw*.15, row*ch+ch*.15
        x2, y2 = (col+1)*cw-cw*.15, (row+1)*ch-ch*.15
        draw.line((x1,y1,x2,y2), fill="red", width=stroke)
        draw.line((x1,y2,x2,y1), fill="red", width=stroke)
    return out

# =============================================================================
# 스도쿠 로직
# =============================================================================
def valid(board, row, col, num):
    for i in range(9):
        if board[row][i] == num or board[i][col] == num:
            return False
        if board[3*(row//3)+i//3][3*(col//3)+i%3] == num:
            return False
    return True

def candidates(board, row, col):
    return [] if board[row][col] else [n for n in range(1, 10) if valid(board, row, col, n)]

def best_empty(board):
    result, best = None, None
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                items = candidates(board, r, c)
                if not items: return r, c, []
                if best is None or len(items) < len(best):
                    result, best = (r, c), items
    return None if result is None else (result[0], result[1], best)

def solve(board):
    empty = best_empty(board)
    if empty is None: return True
    r, c, nums = empty
    for n in nums:
        board[r][c] = n
        if solve(board): return True
        board[r][c] = 0
    return False

def solution_count(board, limit=2):
    empty = best_empty(board)
    if empty is None: return 1
    r, c, nums = empty
    total = 0
    for n in nums:
        board[r][c] = n
        total += solution_count(board, limit)
        board[r][c] = 0
        if total >= limit: return total
    return total

def create_puzzle(level):
    clue_count = {"초급":38, "중급":30, "고급":24}[level]
    solution = [[0]*9 for _ in range(9)]
    solve(solution)
    puzzle = [row[:] for row in solution]
    positions = [(r,c) for r in range(9) for c in range(9)]
    random.shuffle(positions)
    left = 81
    for r, c in positions:
        if left <= clue_count: break
        before = puzzle[r][c]
        puzzle[r][c] = 0
        if solution_count([row[:] for row in puzzle]) == 1:
            left -= 1
        else:
            puzzle[r][c] = before
    return puzzle, solution

def rule_errors(board):
    errors = set()
    def duplicates(cells):
        found = {}
        for r, c in cells:
            if board[r][c]: found.setdefault(board[r][c], []).append((r,c))
        for positions in found.values():
            if len(positions) > 1: errors.update(positions)
    for r in range(9): duplicates([(r,c) for c in range(9)])
    for c in range(9): duplicates([(r,c) for r in range(9)])
    for br in range(0,9,3):
        for bc in range(0,9,3):
            duplicates([(r,c) for r in range(br,br+3) for c in range(bc,bc+3)])
    return errors

def immediate_hints(board):
    hints = set()
    # 후보가 하나뿐인 빈칸
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0 and len(candidates(board,r,c)) == 1:
                hints.add((r,c))
    # 행, 열, 3x3 박스에서 특정 숫자가 들어갈 수 있는 칸이 하나인 경우
    groups = []
    groups += [[(r,c) for c in range(9)] for r in range(9)]
    groups += [[(r,c) for r in range(9)] for c in range(9)]
    groups += [[(r,c) for r in range(br,br+3) for c in range(bc,bc+3)] for br in range(0,9,3) for bc in range(0,9,3)]
    for group in groups:
        places = {n:[] for n in range(1,10)}
        for r,c in group:
            if board[r][c] == 0:
                for n in candidates(board,r,c): places[n].append((r,c))
        for cells in places.values():
            if len(cells) == 1: hints.add(cells[0])
    return hints

# =============================================================================
# 보관함
# =============================================================================
def load_records(level=None):
    try:
        records = json.loads(DB_FILE.read_text(encoding="utf-8")) if DB_FILE.exists() else []
        return [x for x in records if x.get("difficulty") == level] if level else records
    except (OSError, json.JSONDecodeError):
        return []

def save_record(level, puzzle, solution):
    records = load_records()
    records.append({"id":max([x.get("id",0) for x in records], default=0)+1, "difficulty":level, "puzzle":puzzle, "solution":solution, "created_at":dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
    temp = DB_FILE.with_suffix(".tmp")
    temp.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(DB_FILE)

# =============================================================================
# 표, PNG, A4 PDF
# =============================================================================
def board_html(board, solution=None, errors=None, hints=None):
    errors, hints = errors or set(), hints or set()
    html = "<div class='sudoku-wrap'><table class='sudoku'>"
    for r in range(9):
        html += "<tr>"
        for c in range(9):
            value = board[r][c]
            cls = "error" if (r,c) in errors else "hint" if (r,c) in hints else "answer" if value == 0 and solution else ""
            text = value or (solution[r][c] if solution else "")
            html += f"<td class='{cls}'>{text}</td>"
        html += "</tr>"
    return html + "</table></div>"

def font(size, bold=False):
    paths = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"]
    for path in paths:
        if os.path.exists(path): return ImageFont.truetype(path, size)
    return ImageFont.load_default()

def board_png(board, title="Daily Sudoku Puzzle"):
    cell, margin, title_h = 72, 42, 72
    size = cell * 9
    img = Image.new("RGB", (size+margin*2, size+margin*2+title_h), "white")
    draw = ImageDraw.Draw(img)
    draw.text((margin,18), title, fill="#111827", font=font(28, True))
    x0, y0 = margin, margin+title_h
    for i in range(10):
        width = 5 if i % 3 == 0 else 1
        draw.line((x0+i*cell,y0,x0+i*cell,y0+size), fill="#111", width=width)
        draw.line((x0,y0+i*cell,x0+size,y0+i*cell), fill="#111", width=width)
    num_font = font(38, True)
    for r in range(9):
        for c in range(9):
            if board[r][c]:
                box = draw.textbbox((0,0),str(board[r][c]),font=num_font)
                draw.text((x0+c*cell+(cell-(box[2]-box[0]))/2, y0+r*cell+(cell-(box[3]-box[1]))/2-4), str(board[r][c]), fill="#111", font=num_font)
    out = io.BytesIO(); img.save(out, format="PNG", optimize=True); return out.getvalue()

def puzzle_pdf(board, date_value, difficulty):
    out = io.BytesIO()
    pdf = canvas.Canvas(out, pagesize=A4)

    page_width, page_height = A4
    board_size = 160 * mm
    cell_size = board_size / 9
    left = (page_width - board_size) / 2
    bottom = 57 * mm

    # 제목
    pdf.setFillColor(HexColor("#111827"))
    pdf.setFont("Helvetica-Bold", 26)
    pdf.drawCentredString(
        page_width / 2,
        page_height - 30 * mm,
        "Daily Sudoku Puzzle",
    )

    # 날짜
    pdf.setFillColor(HexColor("#4B5563"))
    pdf.setFont("Helvetica", 12)
    pdf.drawCentredString(
        page_width / 2,
        page_height - 39 * mm,
        date_value.strftime("%Y.%m.%d"),
    )

    # 난이도: 초급 1칸 / 중급 2칸 / 고급 3칸
    difficulty_count = {"초급": 1, "중급": 2, "고급": 3,}.get(difficulty, 1)
    square_size = 5 * mm
    square_gap = 2 * mm
    total_width = (difficulty_count * square_size + (difficulty_count - 1) * square_gap)

    start_x = (page_width - total_width) / 2
    square_y = page_height - 49 * mm

    pdf.setFillColor(HexColor("#2563EB"))

    for index in range(difficulty_count):
    square_x = start_x + index * (square_size + square_gap)

    pdf.roundRect( square_x,square_y,square_size,square_size,1.2 * mm,
        stroke=0,
        fill=1,
    )

    # 스도쿠 격자
    pdf.setStrokeColor(HexColor("#111111"))

    for index in range(10):
        pdf.setLineWidth(2.1 if index % 3 == 0 else 0.45)
        position = index * cell_size

        pdf.line(
            left + position,
            bottom,
            left + position,
            bottom + board_size,
        )

        pdf.line(
            left,
            bottom + position,
            left + board_size,
            bottom + position,
        )

    # 주어진 숫자
    pdf.setFillColor(HexColor("#111111"))
    pdf.setFont("Helvetica-Bold", 19)

    for row in range(9):
        for col in range(9):
            value = board[row][col]

            if value == 0:
                continue

            text = str(value)

            x = (
                left
                + col * cell_size
                + (cell_size - stringWidth(text, "Helvetica-Bold", 19)) / 2
            )

            y = bottom + (8 - row) * cell_size + cell_size * 0.31

            pdf.drawString(x, y, text)

    # 하단 문구
    pdf.setFillColor(HexColor("#6B7280"))
    pdf.setFont("Helvetica", 9)
    pdf.drawCentredString(
        page_width / 2,
        24 * mm,
        "Solve one square at a time. Enjoy your puzzle!",
    )

    pdf.save()
    return out.getvalue()
    
def downloads(board, level, prefix):
    st.caption("다운로드 버튼을 누르면 현재 기기(휴대폰 또는 PC)에 파일이 저장됩니다.")
    date_value = st.date_input("인쇄 날짜", value=dt.date.today(), key=prefix+"_date")
    stamp = date_value.strftime("%Y%m%d")
    a,b = st.columns(2)
    a.download_button("🖨️ A4 PDF 저장", puzzle_pdf(board, date_value, level), f"daily_sudoku_{stamp}_{level}.pdf", "application/pdf", key=prefix + "_pdf", use_container_width=True,)
    b.download_button("🖼️ PNG 저장", board_png(board, f"Daily Sudoku Puzzle · {level}"), f"daily_sudoku_{stamp}.png", "image/png", key=prefix+"_png", use_container_width=True)

# =============================================================================
# 화면: 사진 읽기
# =============================================================================
tab_read, tab_make = st.tabs(["📸 사진 읽기 & 확인", "🎲 문제 만들기 & 보관함"])
with tab_read:
    st.subheader("1. 스도쿠 사진 가져오기")
    upload = st.file_uploader("스도쿠 사진을 촬영하거나 업로드하세요", type=["jpg","jpeg","png"])
    if upload:
        file_hash = hashlib.sha256(upload.getvalue()).hexdigest()
        if st.session_state.get("file_hash") != file_hash:
            st.session_state.update({"file_hash":file_hash,"angle":0})
            for k in ["crop_hash","analysis","analysis_img","celebrate"]: st.session_state.pop(k,None)
        try:
            original = normalize(Image.open(upload))
        except Exception:
            st.error("이미지를 열 수 없습니다."); st.stop()
        st.session_state.setdefault("angle",0)
        c1,c2=st.columns(2)
        if c1.button("🔄 90° 회전"):
            st.session_state["angle"]=(st.session_state["angle"]-90)%360; st.session_state.pop("analysis",None)
        if c2.button("↩️ 방향 초기화"):
            st.session_state["angle"]=0; st.session_state.pop("analysis",None)
        work=resize(original)
        if st.session_state["angle"]: work=work.rotate(st.session_state["angle"],expand=True)
        crop=st.checkbox("✂️ 빨간 박스로 9×9 영역 자르기",value=True)
        target=st_cropper(work,realtime_update=True,box_color="#FF0000",aspect_ratio=(1,1),key="cropper") if crop else work
        if target:
            current_hash=digest_image(target)
            if st.session_state.get("crop_hash") != current_hash:
                st.session_state["crop_hash"]=current_hash
                for k in ["analysis","analysis_img","celebrate"]: st.session_state.pop(k,None)
            st.image(target,caption="AI가 읽을 최종 9×9 영역",use_container_width=True)
            key=get_api_key(); model=st.text_input("Gemini 모델",value=DEFAULT_MODEL)
            if not key: st.info("Streamlit Secrets에 GEMINI_API_KEY를 설정해 주세요.")
            elif st.button("🔎 손글씨 읽기 및 정답 확인",type="primary"):
                try:
                    with st.spinner("손글씨를 읽고 스도쿠를 확인하고 있습니다..."):
                        result=read_sudoku_with_ai(gemini_client(key),target,model)
                    st.session_state["analysis"]=result.model_dump(); st.session_state["analysis_img"]=target.copy()
                except Exception as e:
                    st.error("AI 분석 실패: API 키, 모델명, 모델 사용 권한을 확인해 주세요."); st.exception(e)
            if "analysis" in st.session_state:
                grid=validate_grid(SudokuAnalysis.model_validate(st.session_state["analysis"]).grid)
                errors=rule_errors(grid); complete=all(n != 0 for row in grid for n in row)
                st.markdown("---"); st.subheader("🔎 AI가 읽은 9×9 스도쿠 판")
                if errors:
                    st.markdown(board_html(grid,errors=errors),unsafe_allow_html=True)
                    st.error("규칙에 맞지 않는 숫자를 빨간색으로 표시했습니다.")
                    if "analysis_img" in st.session_state: st.image(mark_photo_errors(st.session_state["analysis_img"],errors),caption="원본 사진의 오류 위치",use_container_width=True)
                elif complete:
                    st.markdown(board_html(grid),unsafe_allow_html=True)
                    st.success("🎉 정답입니다! 모든 행, 열, 3×3 박스가 규칙을 만족합니다.")
                    done_hash=hashlib.sha256(json.dumps(grid).encode()).hexdigest()
                    if st.session_state.get("celebrate") != done_hash:
                        st.session_state["celebrate"]=done_hash; st.balloons()
                else:
                    hints=immediate_hints(grid)
                    st.markdown(board_html(grid,hints=hints),unsafe_allow_html=True)
                    st.info("💡 노란색 칸은 현재 상태에서 바로 해결할 수 있는 빈칸입니다." if hints else "현재는 바로 확정할 수 있는 빈칸을 찾지 못했습니다.")
                st.download_button("🖼️ 인식된 9×9 판 PNG 저장",board_png(grid,"AI Read Sudoku Grid"),"recognized_sudoku_grid.png","image/png",key="read_png",use_container_width=True)

# =============================================================================
# 화면: 문제 생성 / 보관함
# =============================================================================
with tab_make:
    st.subheader("🎲 난이도별 스도쿠 문제 생성")
    a,b=st.columns([2,1])
    level=a.selectbox("난이도",["초급","중급","고급"])
    b.write(""); b.write("")
    if b.button("문제 생성",type="primary",use_container_width=True):
        with st.spinner("유일한 정답을 가진 문제를 만들고 있습니다..."):
            puzzle,answer=create_puzzle(level)
        st.session_state.update({"puzzle":puzzle,"answer":answer,"level":level})
        try: save_record(level,puzzle,answer); st.success("문제를 만들고 보관함에 저장했습니다.")
        except OSError: st.warning("문제는 생성됐지만 보관함 저장에는 실패했습니다.")
    if "puzzle" in st.session_state:
        st.markdown(f"### 📋 생성된 문제 · {st.session_state['level']}")
        show=st.toggle("🔍 정답 보기",key="new_sol")
        st.markdown(board_html(st.session_state["puzzle"],st.session_state["answer"] if show else None),unsafe_allow_html=True)
        downloads(st.session_state["puzzle"],st.session_state["level"],"new")
    st.markdown("---"); st.subheader("📁 저장된 문제 보관함")
    choice=st.radio("조회 난이도",["전체","초급","중급","고급"],horizontal=True)
    records=load_records(None if choice=="전체" else choice)
    if not records: st.info("저장된 문제가 없습니다.")
    else:
        idx=st.selectbox("불러올 문제",range(len(records)),format_func=lambda i:f"#{records[i].get('id','?')} [{records[i].get('difficulty','')}] {records[i].get('created_at','')}")
        item=records[idx]; saved=item["puzzle"]; answer=item.get("solution")
        if not answer:
            answer=[r[:] for r in saved]; solve(answer)
        show=st.toggle("🔍 저장된 문제 정답 보기",key="saved_sol")
        st.markdown(board_html(saved,answer if show else None),unsafe_allow_html=True)
        downloads(saved,item.get("difficulty","스도쿠"),f"saved_{item.get('id',idx)}")
