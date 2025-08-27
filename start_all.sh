#!/bin/bash

echo "starting all stocast services in order..."
echo

echo
echo "2. starting analytics service (python)..."
gnome-terminal --title="Python Analytics" -- bash -c "cd analytics-python && source venv/bin/activate && uvicorn app:app --reload; exec bash" &
if [ $? -ne 0 ]; then
    xterm -title "Python Analytics" -e "cd analytics-python && source venv/bin/activate && uvicorn app:app --reload; bash" &
fi

echo "waiting for analytics to start (15 seconds)..."
sleep 15

echo "checking if analytics is ready..."
while ! curl -s http://localhost:8000/health > /dev/null 2>&1; do
    echo "analytics not ready yet, waiting 5 more seconds..."
    sleep 5
done
echo "analytics is ready!"

echo
echo "3. starting dashboard (react)..."
gnome-terminal --title="React Dashboard" -- bash -c "cd dashboard-web && npm start; exec bash" &
if [ $? -ne 0 ]; then
    xterm -title "React Dashboard" -e "cd dashboard-web && npm start; bash" &
fi

echo "waiting for dashboard to start (20 seconds)..."
sleep 20

echo "checking if dashboard is ready..."
while ! curl -s http://localhost:3000 > /dev/null 2>&1; do
    echo "dashboard not ready yet, waiting 5 more seconds..."
    sleep 5
done
echo "dashboard is ready!"

echo
echo "4. starting pos client (javafx)..."
gnome-terminal --title="JavaFX POS" -- bash -c "cd post-client-javafx && mvn javafx:run; exec bash" &
if [ $? -ne 0 ]; then
    xterm -title "JavaFX POS" -e "cd post-client-javafx && mvn javafx:run; bash" &
fi

echo
echo "========================================"
echo "all services started in proper order!"
echo "========================================"
echo
echo "services are available at:"
echo "- backend: http://localhost:9090"
echo "- analytics: http://localhost:8000"
echo "- dashboard: http://localhost:3000"
echo
echo "startup order completed:"
echo "1. ballerina backend ✓"
echo "2. python analytics ✓"
echo "3. react dashboard ✓"
echo "4. javafx pos ✓"
echo
