@echo off
echo Creating configuration file...

(
    echo host = "db.jknffzcmoojysdecocbj.supabase.co"
    echo username = "postgres"
    echo password = "Skynettheone#1"
    echo database = "postgres"
    echo port = 5432
    echo.
    echo [sendgrid]
    echo apiKey="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"
    echo.
    echo [twilio]
    echo accountSid="AC2079dd1befacf9bde1be5095d3106344"
    echo authToken="f296bb1a0911835533d6012ade449265"
    echo senderPhoneNumber="+18564859952"
    echo.
    echo # enableNightlySegmentation = true
    echo # segmentationIntervalHours = 24
) > Config.toml

echo Configuration created. Starting Ballerina service...
bal run