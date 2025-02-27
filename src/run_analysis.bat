@echo off
echo Starting Air Quality Analysis...
echo ==============================

:: Check Python environment
python --version
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    exit /b 1
)

:: Create virtual environment if it doesn't exist
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate.bat
)

:: Run tests first
echo.
echo Running validation tests...
echo ------------------------------
python src/test_updates.py
if errorlevel 1 (
    echo Error: Tests failed
    echo Please fix the issues before proceeding
    pause
    exit /b 1
)

:: Create required directories
mkdir models 2>nul
mkdir logs 2>nul
mkdir results 2>nul
mkdir preprocessed 2>nul

echo.
echo Environment ready. Starting notebook validation...
echo ------------------------------

:: Validate notebooks
python src/validate_notebooks.py
if errorlevel 1 (
    echo Error: Notebook validation failed
    pause
    exit /b 1
)

echo.
echo Combining notebooks...
echo ------------------------------

:: Combine notebooks
python src/combine_notebooks.py
if errorlevel 1 (
    echo Error: Notebook combination failed
    pause
    exit /b 1
)

echo.
echo Analysis Pipeline:
echo 1. Data Preprocessing
echo 2. Model Definitions [including CNN-LSTM and CNN-BiLSTM hybrids]
echo 3. Model Training
echo 4. Performance Analysis [with efficiency metrics]
echo 5. Model Optimization
echo 6. Temperature and Humidity Prediction
echo 7. Visualization

echo.
echo To run the analysis:
echo 1. Start Jupyter Notebook:
echo    jupyter notebook src/final_air_quality_analysis.ipynb
echo.
echo 2. Run all cells in order
echo.
echo 3. Check results in:
echo    - models/ for saved models
echo    - results/ for metrics and comparisons
echo    - logs/ for training logs
echo.
echo Note: Look for efficiency comparisons between
echo standard and hybrid models in the results.

:: Keep terminal open
pause