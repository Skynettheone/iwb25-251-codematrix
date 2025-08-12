import ballerina/http;
import ballerina/log;

listener http:Listener httpListener = check new (9090);

service /api on httpListener {

    resource function get status() returns json {
        log:printInfo("Request received for backend status check.");
        return {"status": "Ballerina backend is running successfully"};
    }
}