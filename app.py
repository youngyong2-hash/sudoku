import os
import io
import json
import random
import hashlib
import datetime as dt
from pathlib import Path

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
# 설정
# =============================================================================
st.set_page_config(page_title="🏄영용's Sudoku", page_icon="🏄", layout="centered")
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

def ai_read_sudoku(client, image, model_name):
    instruction = """
당신은 스도쿠 사진 판독 전문 AI입니다. 사진 속 9x9 스도쿠를 읽으세요.
인쇄 숫자와 손글씨 숫자를 모두 읽어 grid에 기록하세요.
- grid는 9개의 행과 각 행의 9개 숫자로 구성됩니다.
- 빈칸 또는 확신할 수 없는 숫자는 0으로 기록하고 추측하지 마세요.
- errors와 single_hint도 스키마에 맞게 반환하세요.
- 격자 밖의 텍스트와 메모는 무시하세요.
"""
    response = client.models.generate_content(
        model=model_name,
        contents=[image, "사진의 스도쿠를 읽어 9x9 JSON 데이터로 반환하세요."],
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

def generate_puzzle(difficulty):
    needed = {"초급":38, "중급":30, "고급":24}[difficulty]
    answer = [[0] * 9 for _ in range(9)]
    solve(answer)
    puzzle = [row[:] for row in answer]
    positions = [(row, col) for row in range(9) for col in range(9)]
    random.shuffle(positions)
    left = 81
    for row, col in positions:
        if left <= needed:
            break
        old = puzzle[row][col]
        puzzle[row][col] = 0
        if count_solutions([r[:] for r in puzzle]) == 1:
            left -= 1
        else:
            puzzle[row][col] = old
    return puzzle, answer

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
# 보관함
# =============================================================================
def load_puzzles(difficulty=None):
    try:
        items = json.loads(DB_FILE.read_text(encoding="utf-8")) if DB_FILE.exists() else []
        return [item for item in items if item.get("difficulty") == difficulty] if difficulty else items
    except (OSError, json.JSONDecodeError):
        return []

def save_puzzle(difficulty, puzzle, answer):
    items = load_puzzles()
    items.append({"id":max([item.get("id",0) for item in items], default=0)+1,"difficulty":difficulty,"puzzle":puzzle,"solution":answer,"created_at":dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
    tmp = DB_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(DB_FILE)

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
    data = io.BytesIO(); pdf = canvas.Canvas(data,pagesize=A4)
    width,height = A4; size = 160*mm; cell = size/9; left = (width-size)/2; bottom = 57*mm
    pdf.setFillColor(HexColor("#111827")); pdf.setFont("Helvetica-Bold",26); pdf.drawCentredString(width/2,height-30*mm,"Daily Sudoku Puzzle")
    pdf.setFillColor(HexColor("#4B5563")); pdf.setFont("Helvetica",12); pdf.drawCentredString(width/2,height-39*mm,date_value.strftime("%Y.%m.%d"))
    count = {"초급":1,"중급":2,"고급":3}.get(difficulty,1); square=5*mm; gap=2*mm
    start=(width-(count*square+(count-1)*gap))/2; y=height-49*mm; pdf.setFillColor(HexColor("#2563EB"))
    for index in range(count):
        pdf.roundRect(start+index*(square+gap),y,square,square,1.2*mm,stroke=0,fill=1)
    pdf.setStrokeColor(HexColor("#111111"))
    for index in range(10):
        pdf.setLineWidth(2.1 if index%3==0 else .45); p=index*cell
        pdf.line(left+p,bottom,left+p,bottom+size); pdf.line(left,bottom+p,left+size,bottom+p)
    pdf.setFillColor(HexColor("#111111")); pdf.setFont("Helvetica-Bold",19)
    for row in range(9):
        for col in range(9):
            if board[row][col]:
                text=str(board[row][col]); x=left+col*cell+(cell-stringWidth(text,"Helvetica-Bold",19))/2
                pdf.drawString(x,bottom+(8-row)*cell+cell*.31,text)
    pdf.setFillColor(HexColor("#6B7280")); pdf.setFont("Helvetica",9); pdf.drawCentredString(width/2,24*mm,"Solve one square at a time. Enjoy your puzzle!")
    pdf.save(); return data.getvalue()

def download_buttons(board,difficulty,prefix):
    st.caption("PDF와 PNG는 Google Drive가 아니라 현재 사용 중인 기기에 저장됩니다.")
    date_value=st.date_input("인쇄 날짜",value=dt.date.today(),key=prefix+"_date"); stamp=date_value.strftime("%Y%m%d")
    left,right=st.columns(2)
    left.download_button("🖨️ A4 PDF 저장",make_pdf(board,date_value,difficulty),f"daily_sudoku_{stamp}.pdf","application/pdf",key=prefix+"_pdf",use_container_width=True)
    right.download_button("🖼️ PNG 저장",make_png(board,f"Daily Sudoku Puzzle · {difficulty}"),f"daily_sudoku_{stamp}.png","image/png",key=prefix+"_png",use_container_width=True)

# =============================================================================
# 탭 1: 모바일 확대/이동/자르기 및 AI 판독
# =============================================================================
tab_read,tab_create=st.tabs(["📸 사진 읽기 & 확인","🎲 문제 만들기 & 보관함"])
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
    first,second=st.columns([2,1])
    difficulty=first.selectbox("난이도",["초급","중급","고급"])
    second.write(""); second.write("")
    if second.button("문제 생성",type="primary",use_container_width=True):
        with st.spinner("유일한 정답을 가진 문제를 만들고 있습니다..."):
            puzzle,answer=generate_puzzle(difficulty)
        st.session_state.update({"puzzle":puzzle,"answer":answer,"difficulty":difficulty})
        try:
            save_puzzle(difficulty,puzzle,answer); st.success("문제를 만들고 보관함에 저장했습니다.")
        except OSError:
            st.warning("문제는 생성됐지만 보관함 저장에는 실패했습니다.")
    if "puzzle" in st.session_state:
        st.markdown(f"### 📋 생성된 문제 · {st.session_state['difficulty']}")
        show=st.toggle("🔍 정답 보기",key="current_solution")
        st.markdown(render_board(st.session_state["puzzle"],st.session_state["answer"] if show else None),unsafe_allow_html=True)
        download_buttons(st.session_state["puzzle"],st.session_state["difficulty"],"new")
    st.markdown("---"); st.subheader("📁 저장된 문제 보관함")
    selected=st.radio("조회 난이도",["전체","초급","중급","고급"],horizontal=True)
    items=load_puzzles(None if selected=="전체" else selected)
    if not items:
        st.info("저장된 문제가 없습니다.")
    else:
        index=st.selectbox("불러올 문제",range(len(items)),format_func=lambda i:f"#{items[i].get('id','?')} [{items[i].get('difficulty','')}] {items[i].get('created_at','')}")
        item=items[index]; saved=item["puzzle"]; solution=item.get("solution")
        if not solution:
            solution=[row[:] for row in saved]; solve(solution)
        show=st.toggle("🔍 저장된 문제 정답 보기",key="saved_solution")
        st.markdown(render_board(saved,solution if show else None),unsafe_allow_html=True)
        download_buttons(saved,item.get("difficulty","초급"),f"saved_{item.get('id',index)}")
