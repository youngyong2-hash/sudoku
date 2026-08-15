from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

URL = "https://sudoku-6lzzbwyx5nrbmml5awpuqg.streamlit.app/"

options = Options()
options.add_argument("--headless=new")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome(options=options)
driver.get(URL)

wait = WebDriverWait(driver, 30)
try:
    wake_btn = wait.until(
        EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'get this app back up')]"))
    )
    wake_btn.click()
except TimeoutException:
    pass  # 이미 깨어있으면 통과

wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-testid='stAppViewContainer']")))
driver.quit()
