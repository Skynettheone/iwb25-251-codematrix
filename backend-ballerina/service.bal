import ballerina/http;
import ballerina/log;
import ballerina/sql;
import ballerinax/postgresql;

// data structures
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

type TransactionInfo record {|
    string transaction_id;
    string customer_id;
    decimal total_amount;
    string created_at;
|};

type AnalysisSummary record {|
    decimal total_sales_lkr;
    int transaction_count;
    string message;
|};

type TransactionItemInfo record {|
    int quantity;
    string created_at;
|};

type StockPrediction record {|
    string product_id;
    int forecast_next_7_days;
    string message;
|};

// database configuration
configurable string host = ?;
configurable string username = ?;
configurable string password = ?;
configurable string database = ?;
configurable int port = ?;

//service clients
final postgresql:Client dbClient = check new (
    host = host,
    username = username,
    password = password,
    database = database,
    port = port
);

final http:Client analyticsClient = check new ("http://localhost:8000");


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
        return {"status": "Ballerina backend is running successfully"};
    }

    resource function post transactions(@http:Payload Transaction transactionData) returns json|http:InternalServerError {
        do {
            sql:ExecutionResult result = check dbClient->execute(`
                INSERT INTO transactions (transaction_id, customer_id, total_amount)
                VALUES (${transactionData.transactionId}, ${transactionData.customerId}, ${transactionData.totalAmount})
            `);
            int|string? lastInsertId = result.lastInsertId;
            if lastInsertId is int {
                foreach var item in transactionData.items {
                    _ = check dbClient->execute(`
                        INSERT INTO transaction_items (transaction_record_id, product_id, quantity, price)
                        VALUES (${lastInsertId}, ${item.productId}, ${item.quantity}, ${item.price})
                    `);
                }
            } else {
                log:printError("Failed to retrieve last insert ID for transaction.");
                return <http:InternalServerError>{ body: "Failed to save transaction items." };
            }
        } on fail error e {
            log:printError("Error saving transaction to database: " + e.message());
            return <http:InternalServerError>{ body: "Database error occurred" };
        }
        json response = {
            "message": "Transaction saved successfully",
            "transactionId": transactionData.transactionId
        };
        return response;
    }

    resource function get transactions() returns TransactionInfo[]|http:InternalServerError {
        stream<TransactionInfo, sql:Error?> transactionStream = dbClient->query(`
            SELECT transaction_id, customer_id, total_amount, created_at::text
            FROM transactions ORDER BY created_at DESC LIMIT 10
        `);
        TransactionInfo[] recentTransactions = [];
        do {
            check from var row in transactionStream do { recentTransactions.push(row); };
        } on fail error e {
            log:printError("Error fetching transactions: " + e.message());
            return <http:InternalServerError>{ body: "Database error occurred" };
        }
        return recentTransactions;
    }

    resource function get summary() returns AnalysisSummary|http:InternalServerError {
        stream<TransactionInfo, sql:Error?> transactionStream = dbClient->query(`
            SELECT transaction_id, customer_id, total_amount, created_at::text
            FROM transactions
        `);
        TransactionInfo[] allTransactions = [];
        do {
            check from var row in transactionStream do { allTransactions.push(row); };
        } on fail error e {
            log:printError("Error fetching all transactions for summary: " + e.message());
            return <http:InternalServerError>{ body: "Database error occurred" };
        }
        AnalysisSummary|error summaryResult = analyticsClient->/analyze/summary.post(allTransactions);
        if summaryResult is error {
            log:printError("Error calling analytics service: " + summaryResult.message());
            return <http:InternalServerError>{ body: "Analytics service error occurred" };
        }
        return summaryResult;
    }

    // stock prediction endpoint
    resource function get products/[string productId]/prediction() returns StockPrediction|http:InternalServerError|http:NotFound {
        log:printInfo("Processing prediction request for product: " + productId);

        // fetch historical sales data
        stream<TransactionItemInfo, sql:Error?> historyStream = dbClient->query(`
            SELECT ti.quantity, t.created_at::text
            FROM transaction_items ti
            JOIN transactions t ON ti.transaction_record_id = t.id
            WHERE ti.product_id = ${productId}
            ORDER BY t.created_at
        `);
        
        TransactionItemInfo[] history = [];
        do {
            check from var row in historyStream do { 
                history.push(row); 
            };
        } on fail error e {
            log:printError("Database error while fetching product history: " + e.message());
            return <http:InternalServerError>{ body: "Database error occurred" };
        }

        // validate sufficent data for prediction
        if history.length() < 2 {
            log:printWarn("Insufficient data for prediction - Product: " + productId + ", Records: " + history.length().toString());
            return <http:NotFound>{
                body: {
                    message: "Insufficient historical data for prediction. At least 2 transaction records required."
                }
            };
        }

        log:printInfo("Found " + history.length().toString() + " historical records for product: " + productId);

        json payload = {
            product_id: productId,
            history: history
        };
        
        http:Request request = new;
        request.setJsonPayload(payload);
        request.setHeader("Content-Type", "application/json");
        
        // call analytics service for prediction
        http:Response|error response = analyticsClient->/predict/stock.post(request);
        
        if response is error {
            log:printError("Failed to connect to analytics service: " + response.message());
            return <http:InternalServerError>{ body: "Analytics service unavailable" };
        }
        
        if response.statusCode != 200 {
            log:printError("Analytics service error - Status: " + response.statusCode.toString());
            json|error responseBody = response.getJsonPayload();
            string errorMsg = responseBody is json ? responseBody.toString() : "Unknown error";
            return <http:InternalServerError>{ body: "Prediction failed: " + errorMsg };
        }
        
        json|error jsonPayload = response.getJsonPayload();
        if jsonPayload is error {
            log:printError("Failed to parse analytics response: " + jsonPayload.message());
            return <http:InternalServerError>{ body: "Invalid response from analytics service" };
        }
        
        StockPrediction|error predictionResult = jsonPayload.cloneWithType(StockPrediction);
        if predictionResult is error {
            log:printError("Failed to convert analytics response: " + predictionResult.message());
            return <http:InternalServerError>{ body: "Invalid prediction data format" };
        }
        
        log:printInfo("Successfully generated prediction for product: " + productId);
        return predictionResult;
    }
}
