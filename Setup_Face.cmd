@echo off
setlocal DisableDelayedExpansion
cd /d "%~dp0"
if not exist "runtime\python.exe" (
  echo Run install.cmd first.
  pause
  exit /b 1
)
echo Preparing optional face capture in a separate CPU runtime.
echo This downloads pinned packages and the MediaPipe FaceLandmarker model.
"runtime\python.exe" -I -X utf8 -B "tools\setup_face.py"
set "face_result=%errorlevel%"
pause
exit /b %face_result%
