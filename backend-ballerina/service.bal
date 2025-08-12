import ballerina/http;
import ballerina/log;
import ballerina/sql;
import ballerinax/postgresql;

// --- Data Structures ---
type Item record {|
    string productId;
    int quantity;
    decimal price;
|};

type Transaction record {|
    string transactionId;
    string customerId;
    Item[] items;
    decimal totalAmount;
|};

// --- Database Configuration ---
// Read database connection details from Config.toml
configurable string host = ?;
configurable string username = ?;
configurable string password = ?;
configurable string database = ?;
configurable int port = ?;

final postgresql:Client dbClient = check new (
    host = host,
    username = username,
    password = password,
    database = database,
    port = port
);


// --- HTTP Service ---
listener http:Listener httpListener = check new (9090);

@http:ServiceConfig {
    cors: {
        allowOrigins: ["http://localhost:3000"],
        allowMethods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allowHeaders: ["Content-Type", "Authorization"]
    }
}
service /api on httpListener {

    resource function get status() returns json {
        log:printInfo("Request received for backend status check.");
        return {"status": "Ballerina backend is running successfully"};
    }

    // UPDATED ENDPOINT: Now saves the transaction to the database.
    resource function post transactions(@http:Payload Transaction transactionData) returns json|http:InternalServerError {

        // Use a do block with error handling for database operations
        do {
            // 1. Insert into the main 'transactions' table
            sql:ExecutionResult result = check dbClient->execute(`
                INSERT INTO transactions (transaction_id, customer_id, total_amount)
                VALUES (${transactionData.transactionId}, ${transactionData.customerId}, ${transactionData.totalAmount})
            `);

            // Get the auto-generated ID of the new transaction record
            int|string? lastInsertId = result.lastInsertId;
            if lastInsertId is int {
                // 2. Loop through items and insert them into the 'transaction_items' table
                foreach var item in transactionData.items {
                    _ = check dbClient->execute(`
                        INSERT INTO transaction_items (transaction_record_id, product_id, quantity, price)
                        VALUES (${lastInsertId}, ${item.productId}, ${item.quantity}, ${item.price})
                    `);
                }
            } else {
                // If we don't get an ID, something went wrong.
                log:printError("Failed to retrieve last insert ID for transaction.");
                return <http:InternalServerError>{ body: "Failed to save transaction items." };
            }
        } on fail error e {
            log:printError("Error saving transaction to database: " + e.message());
            return <http:InternalServerError>{ body: "Database error occurred" };
        }

        log:printInfo("Successfully saved transaction to database, transactionId: " + transactionData.transactionId);

        json response = {
            "message": "Transaction saved successfully",
            "transactionId": transactionData.transactionId
        };
        return response;
    }
}
