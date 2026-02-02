@echo off
setlocal enabledelayedexpansion

echo ======================================
echo Starting evaluation script...
echo ======================================

echo Current directory: %CD%
echo.

echo Activating conda environment...
call C:\Users\19715\miniconda3\Scripts\activate.bat biyeshejihuanjing
if errorlevel 1 (
    echo [ERROR] Failed to activate conda environment
    exit /b 1
)
echo [OK] Environment activated
echo.

echo Python path:
where python
echo.

echo Running evaluation script...
python scripts\comprehensive_evaluation.py
if errorlevel 1 (
    echo [ERROR] Evaluation script failed with error code !errorlevel!
    exit /b !errorlevel!
)

echo.
echo ======================================
echo Evaluation completed successfully!
echo ======================================
