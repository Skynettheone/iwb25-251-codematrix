@echo off
echo starting all stocast services in order...
echo.

echo waiting for backend to start (30 seconds)...
timeout /t 30 /nobreak >nul

echo checking if backend is ready...
:check_backend
curl -s http://localhost:9090/health >nul 2>&1
if %errorlevel% neq 0 (
    echo backend not ready yet, waiting 5 more seconds...
    timeout /t 5 /nobreak >nul
    goto check_backend
)
echo backend is ready!

echo.
echo 2. starting analytics service (python)...
start "Python Analytics" cmd /k "cd analytics-python && venv\Scripts\activate.bat && uvicorn app:app --reload"

echo waiting for analytics to start (15 seconds)...
timeout /t 15 /nobreak >nul

echo checking if analytics is ready...
:check_analytics
curl -s http://localhost:8000/health >nul 2>&1
if %errorlevel% neq 0 (
    echo analytics not ready yet, waiting 5 more seconds...
    timeout /t 5 /nobreak >nul
    goto check_analytics
)
echo analytics is ready!

echo.
echo 3. starting dashboard (react)...
start "React Dashboard" cmd /k "cd dashboard-web && npm start"

echo waiting for dashboard to start (20 seconds)...
timeout /t 20 /nobreak >nul

echo checking if dashboard is ready...
:check_dashboard
curl -s http://localhost:3000 >nul 2>&1
if %errorlevel% neq 0 (
    echo dashboard not ready yet, waiting 5 more seconds...
    timeout /t 5 /nobreak >nul
    goto check_dashboard
)
echo dashboard is ready!

echo.
echo 4. starting pos client (javafx)...
start "JavaFX POS" cmd /k "cd post-client-javafx && mvn javafx:run"

echo.
echo ========================================
echo all services started in proper order!
echo ========================================
echo.
echo services are available at:
echo - backend: http://localhost:9090
echo - analytics: http://localhost:8000
echo - dashboard: http://localhost:3000
echo.
echo startup order completed:
echo 1. ballerina backend ✓
echo 2. python analytics ✓
echo 3. react dashboard ✓
echo 4. javafx pos ✓
echo.
pause
