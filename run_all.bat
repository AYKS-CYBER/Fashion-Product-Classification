@echo off
title Fashion Product Classification System Launcher
cd /d "C:\Users\Asus\.gemini\antigravity\scratch\fashion_product_classification"

echo ================================================================
echo    Starting Fashion Product Classifier & Presentation Servers
echo ================================================================

echo 1. Launching Streamlit Web App on Port 8501...
start "Fashion AI App" cmd /k "python -m streamlit run app.py --server.port 8501"

echo 2. Launching Presentation and Report Server on Port 8502...
start "Presentation Server" cmd /k "python -m http.server 8502"

timeout /t 3 >nul

echo 3. Opening Google Chrome...
start http://localhost:8501
start http://localhost:8502/presentation.html
start http://localhost:8502/report.html

echo All services are running!
