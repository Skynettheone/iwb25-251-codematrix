#!/bin/bash
echo "Creating configuration file..."

cat > Config.toml << EOF
host = "db.jknffzcmoojysdecocbj.supabase.co"
username = "postgres"
password = "Skynettheone#1"
database = "postgres"
port = 5432
EOF

echo "Configuration created. Starting Ballerina service..."
bal run
