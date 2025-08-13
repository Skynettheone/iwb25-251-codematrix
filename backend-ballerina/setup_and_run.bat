@echo off
echo Creating configuration file...

(
    echo host = "db.jknffzcmoojysdecocbj.supabase.co"
    echo username = "postgres"
    echo password = "Skynettheone#1"
    echo database = "postgres"
    echo port = 5432
) > Config.toml

echo Configuration created. Starting Ballerina service...
bal run