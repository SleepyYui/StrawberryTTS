@echo off

REM create venv
echo Creating venv...
python -m venv venv

REM set PYTHON to python in venv
set PYTHON=venv\Scripts\python

REM upgrade pip
echo Upgrading pip...
%PYTHON% -m pip install --upgrade pip

REM Checking for espeak
echo Checking if espeak header exists in C:\Program Files\expeak\speak_lib.h
if exist "C:\Program Files\espeakHeaders" (
    echo Found expeak header!
) else (
    echo Could not find expeak header!
    echo Installing espeak from source...
    REM Copying espeak directory to C:\Program Files\espeak
    xcopy /E /H /I /Y espeakWin "C:\Program Files\espeakHeaders"
    echo Copied espeak header!

    REM Checking if folder creation was successful
    echo Checking if folder creation was successful...
    if exist "C:\Program Files\espeakHeaders" (
        echo Folder creation was successful!
    ) else (
        echo Folder creation was not successful!
        echo Please try again! If this error persists, try running this script as administrator or copy the espeakWin directory to C:\Program Files\espeakHeaders
        pause
        exit
    )
)

:: REM Checking if espeak directory is in path
:: echo Checking if espeak directory is in path...
:: setx path "%path%;C:\Program Files\espeakHeaders" /M
:: echo Added espeak directory to path!


REM install requirements
echo Installing requirements...
%PYTHON% -m pip install -r requirements.txt

REM Done?
echo Done installing!