@echo off
setlocal EnableExtensions

rem CMD wrapper for evaluate_all_checkpoints_full_surface.ps1.
rem Override defaults with environment variables before running, e.g.:
rem   set N_PATHS=1000
rem   set FORCE=1
rem   scripts\evaluate_all_checkpoints_full_surface.cmd

set "SCRIPT_DIR=%~dp0"

if not defined PYTHON set "PYTHON=C:\ProgramData\anaconda3\python.exe"
if not defined N_PATHS set "N_PATHS=10000"
if not defined DEVICE set "DEVICE=auto"
if not defined FM_N_STEPS set "FM_N_STEPS=20"
if not defined FM_SOLVER set "FM_SOLVER=euler"
if not defined SIGNATURE_DEPTH set "SIGNATURE_DEPTH=3"

set "FORCE_ARG="
if "%FORCE%"=="1" set "FORCE_ARG=-Force"

set "CALIBRATE_ARG="
if "%CALIBRATE_MOMENTS%"=="1" set "CALIBRATE_ARG=-CalibrateMoments"

set "LIMIT_ARG="
if defined LIMIT set "LIMIT_ARG=-Limit %LIMIT%"

powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%evaluate_all_checkpoints_full_surface.ps1" ^
  -Python "%PYTHON%" ^
  -NPaths %N_PATHS% ^
  -Device "%DEVICE%" ^
  -FmNSteps %FM_N_STEPS% ^
  -FmSolver "%FM_SOLVER%" ^
  -SignatureDepth %SIGNATURE_DEPTH% ^
  %LIMIT_ARG% ^
  %FORCE_ARG% ^
  %CALIBRATE_ARG% ^
  %*

if errorlevel 1 (
  echo Evaluation failed.
  exit /b %errorlevel%
)

echo Evaluation finished.
endlocal
