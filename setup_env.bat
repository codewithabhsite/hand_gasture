@echo off
REM Batch script to set up Python environment and install dependencies

REM Create virtual environment named env
python -m venv env

REM Activate the virtual environment
call env\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip

REM Install requirements
pip install -r requirements.txt

echo.
echo =======================================
echo Setup complete. To activate the environment later, run:
echo     call env\Scripts\activate.bat
echo Then run your Python scripts as needed.
echo =======================================
echo.
pause
