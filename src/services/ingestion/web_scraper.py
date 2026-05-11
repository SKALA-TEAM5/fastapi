# --------------------------------------------------------------------------
# 작성자   : 송상민(ss19801)
# 작성일   : 2026-05-04
#
# [ 주요 함수 정의 ]
#
# 1. fetch_law_data()         : 국가법령정보센터 법령 전문 수집 (requests → Selenium fallback)
# 2. _try_requests()          : requests 기반 법령 수집 시도
# 3. _try_selenium()          : Selenium 기반 법령 수집 시도 (iframe 구조 대응)
# 4. _extract_effective_date(): 법령 시행일 문자열 파싱
# --------------------------------------------------------------------------

import logging
import re
import time
from datetime import date

import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

LAW_URL = (
    "https://www.law.go.kr/"
    "%ED%96%89%EC%A0%95%EA%B7%9C%EC%B9%99/"
    "%EA%B1%B4%EC%84%A4%EC%97%85%EC%82%B0%EC%97%85%EC%95%88%EC%A0%84%EB%B3%B4%EA%B1%B4"
    "%EA%B4%80%EB%A6%AC%EB%B9%84%EA%B3%84%EC%83%81%EB%B0%8F%EC%82%AC%EC%9A%A9%EA%B8%B0%EC%A4%80"
)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}


def _parse_date(text: str) -> date | None:
    for pat in [
        r"(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})",
        r"(\d{4})-(\d{2})-(\d{2})",
        r"(\d{4})(\d{2})(\d{2})",
    ]:
        m = re.search(pat, text)
        if m:
            try:
                return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                continue
    return None


def _extract_effective_date(text: str) -> date | None:
    for line in text.splitlines():
        if re.search(r"시행\s*[일자:]|\b시행\s*\d{4}", line):
            d = _parse_date(line)
            if d:
                return d
    return None


def _try_requests() -> dict | None:
    try:
        resp = requests.get(LAW_URL, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        text = BeautifulSoup(resp.text, "html.parser").get_text(separator="\n", strip=True)
        effective_date = _extract_effective_date(text)
        if effective_date and len(text) > 1000:
            return {"effective_date": str(effective_date), "content": text}
    except Exception as e:
        log.warning(f"requests 실패: {e}")
    return None


def _try_selenium() -> dict | None:
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By

        opts = Options()
        for arg in ("--headless", "--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"):
            opts.add_argument(arg)

        driver = webdriver.Chrome(options=opts)
        try:
            driver.get(LAW_URL)
            time.sleep(4)
            full_text = ""
            for iframe in driver.find_elements(By.TAG_NAME, "iframe"):
                try:
                    driver.switch_to.frame(iframe)
                    full_text += driver.find_element(By.TAG_NAME, "body").text + "\n"
                    driver.switch_to.default_content()
                except Exception:
                    driver.switch_to.default_content()
            if not full_text.strip():
                full_text = driver.find_element(By.TAG_NAME, "body").text
            effective_date = _extract_effective_date(full_text)
            if effective_date:
                return {"effective_date": str(effective_date), "content": full_text}
        finally:
            driver.quit()
    except Exception as e:
        log.warning(f"Selenium 실패: {e}")
    return None


def fetch_law_data() -> dict | None:
    """법령 데이터 취득: requests 우선, 실패 시 Selenium 재시도."""
    result = _try_requests()
    if result:
        log.info("스크래핑 성공 (requests)")
        return result
    log.info("requests 실패 — Selenium 재시도 중...")
    result = _try_selenium()
    if result:
        log.info("스크래핑 성공 (Selenium)")
    return result
