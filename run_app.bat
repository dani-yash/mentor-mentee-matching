@echo off
echo ================================================
echo Setting up the Mentor-Mentee Matching Tool...
echo ================================================

REM Debugging mode: To enable debugging, remove the next line's "REM " prefix
REM @echo on

REM Move to the script's directory
cd /d "%~dp0"

REM Step 1: Ensure requirements.txt exists
if not exist "requirements.txt" (
    echo Error: requirements.txt not found in the current directory.
    echo Please ensure all necessary files app.py, matchAlgo.py, requirements.txt are in the same folder.
    pause
    exit /b
)

REM Step 2: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed on your system. Installing Python...

    REM Download Python installer
    echo Downloading Python installer...
    powershell -Command "& { (New-Object Net.WebClient).DownloadFile('https://www.python.org/ftp/python/3.11.5/python-3.11.5-amd64.exe', 'python_installer.exe') }"
    
    REM Run the Python installer silently
    echo Installing Python silently...
    start /wait python_installer.exe /quiet InstallAllUsers=1 PrependPath=1
    
    REM Remove the installer after installation
    del python_installer.exe
    
    REM Verify if Python installation succeeded
    python --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo Python installation failed. Please install Python manually from https://www.python.org.
        pause
        exit /b
    )
) else (
    echo Python is already installed.
)

REM Step 3: Check if pip is installed
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo pip is not installed. Installing pip...
    python -m ensurepip --upgrade >nul 2>&1
)

REM Step 4: Create and activate the virtual environment
if not exist "venv" (
    echo Creating a virtual environment...
    python -m venv venv
    echo Activating the virtual environment and installing dependencies...
    call venv\Scripts\activate
    pip install --upgrade pip
    pip install -r requirements.txt || pip install streamlit
) else (
    echo Activating the existing virtual environment...
    call venv\Scripts\activate
)

REM Step 5: Check if Streamlit is installed
streamlit --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Streamlit is not installed. Installing Streamlit...
    pip install streamlit
)

REM Step 6: Run the Streamlit app
echo ================================================
echo Starting the Mentor-Mentee Matching Tool...
echo ================================================
start "" streamlit run app.py

REM Keep the terminal open
echo The app is now running. To stop it, close the browser window or press CTRL+C in the terminal.
pause