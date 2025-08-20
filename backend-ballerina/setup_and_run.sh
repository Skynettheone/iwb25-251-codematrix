#!/bin/bash
echo "Creating configuration file..."

cat > Config.toml << EOF
host = "db.jknffzcmoojysdecocbj.supabase.co"
username = "postgres"
password = "Skynettheone#1"
database = "postgres"
port = 5432

[sendgrid]
apiKey="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"

[twilio]
accountSid="AC2079dd1befacf9bde1be5095d3106344"
authToken="f296bb1a0911835533d6012ade449265"
senderPhoneNumber="+18564859952"

# enableNightlySegmentation = true
# segmentationIntervalHours = 24
EOF

echo "Configuration created. Starting Ballerina service..."
bal run
