@echo off
REM Двойной клик: первый раз попросит войти в Threads, дальше собирает и открывает отчёт.
cd /d "%~dp0"
chcp 65001 >nul
python -c "import playwright" 2>nul || (
  echo Устанавливаю Playwright...
  python -m pip install -r requirements.txt && python -m playwright install chromium
)
if not exist browser-profile (
  python scraper.py login
)
python scraper.py collect
start "" "reports\latest.html"
pause
