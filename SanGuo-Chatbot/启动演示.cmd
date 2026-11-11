@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if not errorlevel 1 goto launcher
where python >nul 2>nul
if errorlevel 1 goto missing
python -B -X utf8 web_app.py --port 0 --open-browser
goto finish

:launcher
py -3 -B -X utf8 web_app.py --port 0 --open-browser
goto finish

:missing
echo Python 3.10 or later is required to start SanGuo Chatbot.
pause
exit /b 1

:finish
set "demo_exit=%errorlevel%"
if not "%demo_exit%"=="0" pause
exit /b %demo_exit%
