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
    string category;
    decimal price;
    string description?;
|};

type TransactionView record {|
    string transaction_id;
    string customer_id;
    decimal total_amount;
    string transaction_date;
|};

type SeasonalSales record {|
    string product_id;
    string month;
    int total_quantity;
|};

type ProductCount record {|
    int count;
|};

type Customer record {|
    string customer_id;
    string name;
    string segment;
|};

type SalesData record {|
    string transaction_id;
    string customer_id;
    string created_at;
    string product_id;
    string product_name;
    string category;
    decimal price;
    int quantity;
    decimal price_at_sale;
    decimal line_total;
|};

type DailySalesSummary record {|
    string sale_date;
    string product_id;
    string product_name;
    string category;
    int total_quantity;
    decimal total_revenue;
    int transaction_count;
|};

type PredictionRequest record {|
    string product_id;
    DailySaleInfo[] history;
    int days_to_predict;
    string prediction_type;
|};

// database configuration
configurable string host = ?;
configurable string username = ?;
configurable string password = ?;
configurable string database = ?;
configurable int port = ?;

// service clients
final postgresql:Client dbClient = check new (host = host, username = username, password = password, database = database, port = port);
final http:Client analyticsClient = check new ("http://localhost:8001");

// HTTP service
listener http:Listener httpListener = check new (9090);

@http:ServiceConfig {
    cors: { allowOrigins: ["http://localhost:3000"] }
}
service /api on httpListener {

    resource function get status() returns json {
        return {"status": "Enhanced Ballerina backend is running successfully", "version": "2.0.0"};
    }

    resource function get products() returns Product[]|http:InternalServerError {
        stream<Product, sql:Error?> productStream = dbClient->query(`SELECT product_id, name, category, price FROM products`);
        Product[] products = [];
        
        do {
            check from var row in productStream do { products.push(row); };
        } on fail error e {
            log:printError("DB error fetching products: " + e.message());
            return <http:InternalServerError>{ body: "Database error" };
        }
        
        return products;
    }

    resource function get customers() returns Customer[]|http:InternalServerError {
        stream<Customer, sql:Error?> customerStream = dbClient->query(`SELECT customer_id, name, segment FROM customers ORDER BY name`);
        Customer[] customers = [];
        
        do {
            check from var row in customerStream do { customers.push(row); };
        } on fail error e {
            log:printError("DB error fetching customers: " + e.message());
            return <http:InternalServerError>{ body: "Database error" };
        }
        
        return customers;
    }

    resource function get transactions/latest(int 'limit = 10) returns TransactionView[]|http:InternalServerError {
        stream<TransactionView, sql:Error?> transactionStream = dbClient->query(`
            SELECT transaction_id, customer_id, total_amount, 
                   created_at::text as transaction_date
            FROM transactions 
            ORDER BY created_at DESC 
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

    resource function get products/[string productId]/prediction(int days = 7, string period = "weekly") returns StockPrediction|http:InternalServerError|http:NotFound {
        log:printInfo(string `Processing prediction for product: ${productId}, days: ${days}, period: ${period}`);

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

        PredictionRequest predictionReq = {
            product_id: productId,
            history: history,
            days_to_predict: days,
            prediction_type: period
        };

        StockPrediction|error predictionResult = analyticsClient->/predict/stock.post(predictionReq);
        
        if predictionResult is error {
            log:printError("Analytics service error: " + predictionResult.message());
            return <http:InternalServerError>{ body: "Analytics service error" };
        }
        
        log:printInfo("Successfully generated enhanced prediction for: " + productId);
        return predictionResult;
    }

    resource function post products/predict/all(@http:Payload json requestData) returns StockPrediction[]|http:InternalServerError {
        json|error daysJson = requestData.days;
        json|error periodJson = requestData.period;
        
        int days = daysJson is int ? daysJson : 7;
        string period = periodJson is string ? periodJson : "weekly";
        
        log:printInfo(string `Processing batch predictions for all products: days=${days}, period=${period}`);

        stream<Product, sql:Error?> productStream = dbClient->query(`SELECT product_id, name FROM products`);
        Product[] products = [];
        
        do {
            check from var row in productStream do { products.push(row); };
        } on fail error e {
            log:printError("DB error fetching products: " + e.message());
            return <http:InternalServerError>{ body: "Database error" };
        }

        StockPrediction[] allPredictions = [];
        
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

    resource function get products/seasonal() returns json|http:InternalServerError {
        do {
            // Get seasonal products
            stream<Product, sql:Error?> seasonalStream = dbClient->query(`
                SELECT product_id, name, category, price 
                FROM products 
                WHERE category = 'Seasonal'
                ORDER BY name
            `);
            
            Product[] seasonalProducts = [];
            check from var row in seasonalStream do { seasonalProducts.push(row); };
            
            stream<SeasonalSales, sql:Error?> salesStream = dbClient->query(`
                SELECT 
                    dps.product_id,
                    TO_CHAR(dps.sale_date, 'YYYY-MM') as month,
                    SUM(dps.total_quantity) as total_quantity
                FROM daily_product_sales dps
                JOIN products p ON dps.product_id = p.product_id
                WHERE p.category = 'Seasonal' 
                  AND dps.sale_date >= CURRENT_DATE - INTERVAL '12 months'
                GROUP BY dps.product_id, TO_CHAR(dps.sale_date, 'YYYY-MM')
                ORDER BY month, dps.product_id
            `);
            
            SeasonalSales[] salesData = [];
            check from var row in salesStream do { salesData.push(row); };
            
            json response = {
                "seasonal_products": seasonalProducts,
                "monthly_sales": salesData,
                "analysis_period": "12 months",
                "total_seasonal_products": seasonalProducts.length()
            };
            return response;
        } on fail error e {
            log:printError("DB error fetching seasonal data: " + e.message());
            return <http:InternalServerError>{ body: "Database error" };
        }
    }

    resource function get api/sales/data(string? product_id = (), string? start_date = (), string? end_date = ()) returns SalesData[]|http:InternalServerError {
        string baseQuery = "SELECT t.transaction_id, t.customer_id, t.created_at::text as created_at, " +
                          "ti.product_id, p.name as product_name, p.category, p.price, " +
                          "ti.quantity, ti.price_at_sale, (ti.quantity * ti.price_at_sale) as line_total " +
                          "FROM transactions t " +
                          "JOIN transaction_items ti ON t.id = ti.transaction_record_id " +
                          "JOIN products p ON ti.product_id = p.product_id WHERE 1=1";
        
        string whereClause = "";
        
        if product_id is string {
            whereClause += " AND ti.product_id = '" + product_id + "'";
        }
        
        if start_date is string {
            whereClause += " AND t.created_at >= '" + start_date + "'";
        }
        
        if end_date is string {
            whereClause += " AND t.created_at <= '" + end_date + "'";
        }
        
        string finalQuery = baseQuery + whereClause + " ORDER BY t.created_at";
        
        stream<SalesData, sql:Error?> salesStream = dbClient->query(`${finalQuery}`);
        SalesData[] salesData = [];
        
        do {
            check from var row in salesStream do { salesData.push(row); };
        } on fail error e {
            log:printError("DB error fetching sales data: " + e.message());
            return <http:InternalServerError>{ body: "Database error" };
        }
        
        return salesData;
    }

    resource function get sales/data(string? product_id = ()) returns DailySalesSummary[]|http:InternalServerError {
        string baseQuery = "SELECT dps.sale_date, dps.product_id, p.name as product_name, " +
                          "p.category, dps.total_quantity, " +
                          "(dps.total_quantity * p.price) as total_revenue, " +
                          "1 as transaction_count " +
                          "FROM daily_product_sales dps " +
                          "JOIN products p ON dps.product_id = p.product_id";
        
        string whereClause = "";
        if product_id is string {
            whereClause = " WHERE dps.product_id = '" + product_id + "'";
        }
        
        string finalQuery = baseQuery + whereClause + 
                           " ORDER BY dps.sale_date, dps.product_id";
        
        stream<DailySalesSummary, sql:Error?> summaryStream = dbClient->query(`${finalQuery}`);
        DailySalesSummary[] summaryData = [];
        
        do {
            check from var row in summaryStream do { summaryData.push(row); };
        } on fail error e {
            log:printError("DB error fetching daily sales summary: " + e.message());
            return <http:InternalServerError>{ body: "Database error" };
        }
        
        return summaryData;
    }

    resource function get api/transactions(string? customer_id = (), string? start_date = (), string? end_date = ()) returns TransactionView[]|http:InternalServerError {
        string baseQuery = "SELECT transaction_id, customer_id, total_amount, " +
                          "created_at::text as transaction_date " +
                          "FROM transactions WHERE 1=1";
        
        string whereClause = "";
        
        if customer_id is string {
            whereClause += " AND customer_id = '" + customer_id + "'";
        }
        
        if start_date is string {
            whereClause += " AND created_at >= '" + start_date + "'";
        }
        
        if end_date is string {
            whereClause += " AND created_at <= '" + end_date + "'";
        }
        
        string finalQuery = baseQuery + whereClause + " ORDER BY created_at DESC";
        
        stream<TransactionView, sql:Error?> transactionStream = dbClient->query(`${finalQuery}`);
        TransactionView[] transactions = [];
        
        do {
            check from var row in transactionStream do { transactions.push(row); };
        } on fail error e {
            log:printError("DB error fetching transactions: " + e.message());
            return <http:InternalServerError>{ body: "Database error" };
        }
        
        return transactions;
    }

    resource function get api/products() returns Product[]|http:InternalServerError {
        stream<Product, sql:Error?> productStream = dbClient->query(`SELECT product_id, name, category, price FROM products`);
        Product[] products = [];
        
        do {
            check from var row in productStream do { products.push(row); };
        } on fail error e {
            log:printError("DB error fetching products: " + e.message());
            return <http:InternalServerError>{ body: "Database error" };
        }
        
        return products;
    }

    resource function get api/customers() returns Customer[]|http:InternalServerError {
        stream<Customer, sql:Error?> customerStream = dbClient->query(`SELECT customer_id, name, segment FROM customers ORDER BY name`);
        Customer[] customers = [];
        
        do {
            check from var row in customerStream do { customers.push(row); };
        } on fail error e {
            log:printError("DB error fetching customers: " + e.message());
            return <http:InternalServerError>{ body: "Database error" };
        }
        
        return customers;
    }

    resource function get health() returns json|http:InternalServerError {
        do {
            stream<ProductCount, sql:Error?> result = dbClient->query(`SELECT COUNT(*) as count FROM products`);
            ProductCount[] products = [];
            check from var row in result do { products.push(row); };
            
            json response = {
                "status": "healthy", 
                "database": "connected", 
                "products_count": products.length() > 0 ? products[0].count : 0
            };
            return response;
        } on fail error e {
            log:printError("Database health check failed: " + e.message());
            return <http:InternalServerError>{ body: "Database connection failed" };
        }
    }
}
