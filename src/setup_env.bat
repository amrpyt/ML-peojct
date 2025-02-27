@echo off
echo Creating Python virtual environment...

:: Check if Python is installed
python --version > nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    exit /b 1
)

:: Check if venv exists and remove if it does
if exist .venv (
    echo Removing existing virtual environment...
    rmdir /s /q .venv
)

:: Create new virtual environment
echo Creating new virtual environment...
python -m venv .venv

:: Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

:: Install base requirements first
echo Installing base requirements...
pip install numpy>=1.20.0 pandas>=1.3.0 scikit-learn>=1.0.0 matplotlib>=3.4.0 seaborn>=0.11.0

:: Try installing TensorFlow
echo Attempting to install TensorFlow...
pip install tensorflow>=2.16.0

:: If TensorFlow install fails, try CPU version
if errorlevel 1 (
    echo TensorFlow install failed, trying CPU version...
    pip install tensorflow-cpu>=2.16.0
)

:: Install remaining requirements
echo Installing remaining requirements...
pip install imbalanced-learn>=0.8.0 jupyter>=1.0.0 h5py>=3.10.0

:: Run setup script
echo Running Python setup script...
python src/setup.py

echo.
echo Environment setup complete! To activate the environment:
echo call .venv\Scripts\activate.bat