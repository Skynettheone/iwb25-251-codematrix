#!/bin/bash

echo "========================================"
echo "Stocast - AI-Powered Retail Management"
echo "========================================"
echo

echo "checking prerequisites..."
echo

# check java
if ! command -v java &> /dev/null; then
    echo "ERROR: Java 17+ is required but not found"
    echo "Please install Java 17 or higher from: https://adoptium.net/"
    exit 1
fi

# check node.js
if ! command -v node &> /dev/null; then
    echo "ERROR: Node.js 16+ is required but not found"
    echo "Please install Node.js from: https://nodejs.org/"
    exit 1
fi

# check python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3.8+ is required but not found"
    echo "Please install Python from: https://python.org/"
    exit 1
fi

# check ballerina
if ! command -v bal &> /dev/null; then
    echo "ERROR: Ballerina is required but not found"
    echo "Please install Ballerina from: https://ballerina.io/"
    exit 1
fi

echo "all prerequisites found!"
echo

echo "setting up backend (ballerina)..."
cd backend-ballerina
./setup_and_run.sh
if [ $? -ne 0 ]; then
    echo "ERROR: Backend setup failed"
    exit 1
fi
cd ..

echo
echo "setting up analytics service (python)..."
cd analytics-python
echo "creating virtual environment..."
python3 -m venv venv
source venv/bin/activate
echo "installing dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Python dependencies installation failed"
    exit 1
fi
cd ..

echo
echo "setting up dashboard (react)..."
cd dashboard-web
echo "installing dependencies..."
npm install
if [ $? -ne 0 ]; then
    echo "ERROR: Node.js dependencies installation failed"
    exit 1
fi
cd ..

echo
echo "========================================"
echo "setup completed successfully!"
echo "========================================"
echo
echo "to start the system:"
echo "1. backend: cd backend-ballerina && bal run"
echo "2. analytics: cd analytics-python && python app.py"
echo "3. dashboard: cd dashboard-web && npm start"
echo "4. pos client: cd post-client-javafx && mvn javafx:run"
echo
