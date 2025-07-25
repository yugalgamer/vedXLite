@echo off
REM Launch Flask backend then open browser to localhost automatically
start "" powershell -NoExit -Command "python main.py"
start "" http://localhost:5000/

