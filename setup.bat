@echo off
echo ========================================
echo Stocast - AI-Powered Retail Management
echo ========================================
echo.

echo checking prerequisites...
echo.

REM check java
java -version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Java 17+ is required but not found
    echo Please install Java 17 or higher from: https://adoptium.net/
    pause
    exit /b 1
)

REM check node.js
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Node.js 16+ is required but not found
    echo Please install Node.js from: https://nodejs.org/
    pause
    exit /b 1
)

REM check python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python 3.8+ is required but not found
    echo Please install Python from: https://python.org/
    pause
    exit /b 1
)

REM check ballerina
bal --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Ballerina is required but not found
    echo Please install Ballerina from: https://ballerina.io/
    pause
    exit /b 1
)

echo all prerequisites found!
echo.

echo setting up backend (ballerina)...
cd backend-ballerina
call setup_and_run.bat
if %errorlevel% neq 0 (
    echo ERROR: Backend setup failed
    pause
    exit /b 1
)
cd ..

echo.
echo setting up analytics service (python)...
cd analytics-python
echo creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat
echo installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Python dependencies installation failed
    pause
    exit /b 1
)
cd ..

echo.
echo setting up dashboard (react)...
cd dashboard-web
echo installing dependencies...
npm install
if %errorlevel% neq 0 (
    echo ERROR: Node.js dependencies installation failed
    pause
    exit /b 1
)
cd ..

echo.
echo ========================================
echo setup completed successfully!
echo ========================================
echo.
echo to start the system:
echo 1. backend: cd backend-ballerina && bal run
echo 2. analytics: cd analytics-python && uvicorn app:app --reload
echo 3. dashboard: cd dashboard-web && npm start
echo 4. pos client: cd post-client-javafx && mvn javafx:run
echo.
pause
