@echo off

REM create venv
echo Creating venv...
python -m venv venv

REM set PYTHON to python in venv
set PYTHON=venv\Scripts\python

REM upgrade pip
echo Upgrading pip...
%PYTHON% -m pip install --upgrade pip

REM install requirements
echo Installing requirements...
%PYTHON% -m pip install -r requirements.txt

REM Done?
echo Done installing!