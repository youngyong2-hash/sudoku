import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

URL = "https://sudoku-6lzzbwyx5nrbmml5awpuqg.streamlit.app/"
PASSWORD = os.environ["SUDOKU_PASSWORD"]

options = Options()
options.add_argument("--headless=new")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome(options=options)
driver.get(URL)

wait = WebDriverWait(driver, 30)

# 1) 슬립 상태면 깨우기 버튼 클릭
try:
    wake_btn = wait.until(
        EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'get this app back up')]"))
    )
    wake_btn.click()
except TimeoutException:
    pass  # 이미 켜져 있으면 통과

# 2) 비밀번호 입력창이 뜨면 입력 후 Enter
try:
    pw_input = WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='password']"))
    )
    pw_input.click()
    pw_input.send_keys(PASSWORD)
    pw_input.send_keys(Keys.ENTER)
except TimeoutException:
    pass  # 비밀번호 창이 안 뜨면 통과

driver.quit()
