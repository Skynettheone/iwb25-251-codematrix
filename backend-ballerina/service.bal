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

type DailySaleInfo record {|
    string sale_date;
    int total_quantity;
|};

type StockPrediction record {|
    string product_id;
    int forecast_period_days;
    string prediction_type;
    int total_forecast;
    int[] daily_forecast;
    float[]? weekly_averages;
    float[]? monthly_averages;
    float confidence_score;
    string trend_direction;
    string message;
|};

type Product record {|
    string product_id;
    string name;
    decimal price;
    string description?;
|};

type TransactionView record {|
    string transaction_id;
    string customer_id;
    decimal total_amount;
    string transaction_date;
|};

type PredictionRequest record {|
    string product_id;
    DailySaleInfo[] history;
    int days_to_predict;
    string prediction_type;
|};

// --- Database Configuration ---
configurable string host = ?;
configurable string username = ?;
configurable string password = ?;
configurable string database = ?;
configurable int port = ?;

// --- Service Clients ---
final postgresql:Client dbClient = check new (host = host, username = username, password = password, database = database, port = port);
final http:Client analyticsClient = check new ("http://localhost:8001");

// --- HTTP Service ---
listener http:Listener httpListener = check new (9090);

@http:ServiceConfig {
    cors: { allowOrigins: ["http://localhost:3000"] }
}
service /api on httpListener {

    resource function get status() returns json {
        return {"status": "Enhanced Ballerina backend is running successfully", "version": "2.0.0"};
    }

    // Get all products for selection
    resource function get products() returns Product[]|http:InternalServerError {
        stream<Product, sql:Error?> productStream = dbClient->query(`SELECT product_id, name, price, description FROM products`);
        Product[] products = [];
        
        do {
            check from var row in productStream do { products.push(row); };
        } on fail error e {
            log:printError("DB error fetching products: " + e.message());
            return <http:InternalServerError>{ body: "Database error" };
        }
        
        return products;
    }

    // Get latest transactions for dashboard
    resource function get transactions/latest(int 'limit = 10) returns TransactionView[]|http:InternalServerError {
        stream<TransactionView, sql:Error?> transactionStream = dbClient->query(`
            SELECT transaction_id, customer_id, total_amount, 
                   transaction_date::text as transaction_date
            FROM transactions 
            ORDER BY transaction_date DESC 
            LIMIT ${'limit}
        `);
        
        TransactionView[] transactions = [];
        do {
            check from var row in transactionStream do { transactions.push(row); };
        } on fail error e {
            log:printError("DB error fetching transactions: " + e.message());
            return <http:InternalServerError>{ body: "Database error" };
        }
        
        return transactions;
    }

    // UPDATED & SIMPLIFIED: This endpoint no longer updates the daily_product_sales table.
    // The database trigger you created handles it automatically.
    resource function post transactions(@http:Payload Transaction transactionData) returns json|http:InternalServerError {
        do {
            // 1. Insert into the main transactions table
            sql:ExecutionResult result = check dbClient->execute(`
                INSERT INTO transactions (transaction_id, customer_id, total_amount)
                VALUES (${transactionData.transactionId}, ${transactionData.customerId}, ${transactionData.totalAmount})
            `);
            int|string? lastInsertId = result.lastInsertId;

            if lastInsertId is int {
                // 2. Loop through items and insert them into transaction_items.
                // The database trigger will automatically fire for each insert here.
                foreach var item in transactionData.items {
                    _ = check dbClient->execute(`
                        INSERT INTO transaction_items (transaction_record_id, product_id, quantity, price_at_sale)
                        VALUES (${lastInsertId}, ${item.productId}, ${item.quantity}, ${item.price})
                    `);
                }
            } else {
                return <http:InternalServerError>{ body: "Failed to save transaction items." };
            }
        } on fail error e {
            log:printError("Error saving transaction: " + e.message());
            return <http:InternalServerError>{ body: "Database error occurred" };
        }
        return {"message": "Transaction saved successfully"};
    }

    // Enhanced prediction endpoint with flexible parameters
    resource function get products/[string productId]/prediction(int days = 7, string period = "weekly") returns StockPrediction|http:InternalServerError|http:NotFound {
        log:printInfo(string `Processing prediction for product: ${productId}, days: ${days}, period: ${period}`);

        // Fetch historical data from the pre-aggregated daily_product_sales table
        stream<DailySaleInfo, sql:Error?> historyStream = dbClient->query(`
            SELECT sale_date::text, total_quantity
            FROM daily_product_sales
            WHERE product_id = ${productId}
            ORDER BY sale_date
        `);
        
        DailySaleInfo[] history = [];
        do {
            check from var row in historyStream do { history.push(row); };
        } on fail error e {
            log:printError("DB error fetching product history: " + e.message());
            return <http:InternalServerError>{ body: "Database error" };
        }

        if history.length() < 3 {
            return <http:NotFound>{ body: { message: "Not enough historical data for prediction. At least 3 days required." } };
        }

        // Prepare enhanced request for Python service
        PredictionRequest predictionReq = {
            product_id: productId,
            history: history,
            days_to_predict: days,
            prediction_type: period
        };

        // Call the enhanced Python service
        StockPrediction|error predictionResult = analyticsClient->/predict/stock.post(predictionReq);
        
        if predictionResult is error {
            log:printError("Analytics service error: " + predictionResult.message());
            return <http:InternalServerError>{ body: "Analytics service error" };
        }
        
        log:printInfo("Successfully generated enhanced prediction for: " + productId);
        return predictionResult;
    }

    // Batch prediction for all products
    resource function post products/predict/all(@http:Payload json requestData) returns StockPrediction[]|http:InternalServerError {
        json|error daysJson = requestData.days;
        json|error periodJson = requestData.period;
        
        int days = daysJson is int ? daysJson : 7;
        string period = periodJson is string ? periodJson : "weekly";
        
        log:printInfo(string `Processing batch predictions for all products: days=${days}, period=${period}`);

        // Get all products
        stream<Product, sql:Error?> productStream = dbClient->query(`SELECT product_id, name FROM products`);
        Product[] products = [];
        
        do {
            check from var row in productStream do { products.push(row); };
        } on fail error e {
            log:printError("DB error fetching products: " + e.message());
            return <http:InternalServerError>{ body: "Database error" };
        }

        StockPrediction[] allPredictions = [];
        
        // Generate predictions for each product
        foreach Product product in products {
            stream<DailySaleInfo, sql:Error?> historyStream = dbClient->query(`
                SELECT sale_date::text, total_quantity
                FROM daily_product_sales
                WHERE product_id = ${product.product_id}
                ORDER BY sale_date
            `);
            
            DailySaleInfo[] history = [];
            error? historyResult = from var row in historyStream do { 
                history.push(row); 
            };
            
            if historyResult is error {
                log:printError(string `DB error fetching history for ${product.product_id}: ${historyResult.message()}`);
            } else if history.length() >= 3 {
                PredictionRequest predictionReq = {
                    product_id: product.product_id,
                    history: history,
                    days_to_predict: days,
                    prediction_type: period
                };

                StockPrediction|error predictionResult = analyticsClient->/predict/stock.post(predictionReq);
                
                if predictionResult is StockPrediction {
                    allPredictions.push(predictionResult);
                } else {
                    log:printError(string `Analytics service error for ${product.product_id}: ${predictionResult.message()}`);
                }
            }
        }
        
        log:printInfo(string `Generated ${allPredictions.length()} predictions out of ${products.length()} products`);
        return allPredictions;
    }

    // Initialize sample data
    resource function post initializeData() returns json|http:InternalServerError {
        do {
            // Insert sample products
            _ = check dbClient->execute(`
                INSERT INTO products (product_id, name, price, description) VALUES 
                ('PROD-DHAL-1KG', 'Dal (1KG)', 150.00, 'Premium quality dal'),
                ('PROD-RICE-5KG', 'Rice (5KG)', 800.00, 'Basmati rice'),
                ('PROD-OIL-1L', 'Cooking Oil (1L)', 300.00, 'Sunflower oil'),
                ('PROD-SUGAR-1KG', 'Sugar (1KG)', 120.00, 'White sugar'),
                ('PROD-TEA-250G', 'Tea (250G)', 200.00, 'Black tea')
                ON CONFLICT (product_id) DO NOTHING
            `);

            // Insert sample transactions
            _ = check dbClient->execute(`
                INSERT INTO transactions (transaction_id, customer_id, total_amount, transaction_date) VALUES 
                ('TXN-001', 'CUST-001', 450.00, NOW() - INTERVAL '1 day'),
                ('TXN-002', 'CUST-002', 920.00, NOW() - INTERVAL '2 days'),
                ('TXN-003', 'CUST-003', 320.00, NOW() - INTERVAL '3 days'),
                ('TXN-004', 'CUST-001', 270.00, NOW() - INTERVAL '4 days'),
                ('TXN-005', 'CUST-004', 650.00, NOW() - INTERVAL '5 days')
                ON CONFLICT (transaction_id) DO NOTHING
            `);

            return {"message": "Sample data initialized successfully"};
        } on fail error e {
            log:printError("Failed to initialize sample data: " + e.message());
            return <http:InternalServerError>{ body: "Failed to initialize data" };
        }
    }
}
