import ballerina/http;
import ballerina/log;
import ballerina/sql;
import ballerina/crypto;
import ballerina/uuid;
// Note: time/runtime sleep not available in this Ballerina distribution; retries are immediate.

import ballerinax/postgresql;
// import ballerinax/sendgrid as sgrid;
// import ballerinax/twilio as twil;

// ===== database configuration =====
configurable string host = ?;
configurable string username = ?;
configurable string password = ?;
configurable string database = ?;
configurable int port = ?;

// nightly segmentation scheduler configuration
configurable boolean enableNightlySegmentation = true;
configurable int segmentationIntervalHours = 24;

// segmentation orchestration configuration
// When true, backend will try Python analytics first; otherwise it will use fallback logic only
configurable boolean useAnalyticsForSegmentation = true;
configurable int analyticsSegmentationMaxAttempts = 3;
configurable int analyticsSegmentationInitialDelayMs = 200;

// notification service configurations
type SendgridConfig record {|
    string apiKey;
    string fromEmail?;
|};

configurable SendgridConfig sendgrid = ?;

type TwilioConfig record {|
    string accountSid;
    string authToken;
    string senderPhoneNumber;
|};

configurable TwilioConfig twilio = ?;

final postgresql:Client dbClient = check new (
    host = host,
    username = username,
    password = password,
    database = database,
    port = port,
    options = {
        ssl: {
            mode: postgresql:REQUIRE
        }
    }
);

// ===== service clients =====
// analytics service http client
final http:Client analyticsClient = check new ("http://localhost:8000");

// ===== record definitions =====

type Product record {|
    string product_id;
    string name;
    string category;
    decimal price;
|};

type Customer record {|
    string customer_id;
    string name;
    string? email;
    string? phone_number;
    string? segment;
|};

type Transaction record {|
    string transaction_id;
    string customer_id;
    decimal total_amount;
    string created_at;
|};

type TransactionItem record {|
    string product_id;
    int quantity;
    decimal price_at_sale;
|};

type NewTransactionPayload record {|
    string transaction_id;
    string customer_id;
    decimal total_amount;
    TransactionItem[] items;
|};

type MarketingCampaign record {|
    int id;
    string name;
    string message_template;
    string target_segment;
    string campaign_type;
    string? email_subject;
    string? photo_url;
    string created_at;
|};

type NewCampaignPayload record {|
    string name;
    string message_template;
    string target_segment;
    string campaign_type;
    string? email_subject;
    string? photo_url;
|};

type NewProductPayload record {|
    string product_id;
    string name;
    string category;
    decimal price;
|};

type NewCustomerPayload record {|
    string customer_id;
    string name;
    string? email;
    string? phone_number;
|};

type CustomerSegmentUpdate record {|
    string customer_id;
    string Segment;
|};

type AnalyticsResponse record {| 
    CustomerSegmentUpdate[] data;
|};

// normalize any incoming segment names to the allowed set: New, Champion, Loyal, At-Risk
function normalizeSegmentName(string seg) returns string {
    string s = seg.toLowerAscii();
    if s == "champion" {
        return "Champion";
    } else if s == "loyal" {
        return "Loyal";
    } else if s == "at-risk" || s == "atrisk" || s == "needs attention" || s == "needs_attention" {
        // map analytics "Needs Attention" to "At-Risk"
        return "At-Risk";
    } else if s == "potential loyalist" || s == "potential_loyalist" {
        // collapse potential loyalist into Loyal for the 4-bucket scheme
        return "Loyal";
    } else if s == "regular" {
        // legacy label -> map to New in 4-bucket scheme
        return "New";
    } else if s == "new" {
        return "New";
    } else {
        // default safe bucket
        return "New";
    }
}

// ===== reusable segmentation logic =====
function performCustomerSegmentation() returns json|error {
    log:printInfo("Starting customer segmentation...");
    
    // Get all customers
    stream<record {| string customer_id; string name; string? email; string? phone_number; |}, sql:Error?> customerStream = dbClient->query(`SELECT customer_id, name, email, phone_number FROM customers`);
    record {| string customer_id; string name; string? email; string? phone_number; |}[] customers = check from var row in customerStream select row;
    
    if customers.length() == 0 { 
        return { status: "success", message: "No customers found for segmentation." }; 
    }
    
    // Try to call analytics service to compute RFM/KMeans segments for all customers in one call.
    // If analytics service is unavailable or returns an error, fall back to the legacy frequency-only method.
    int updatedCount = 0;
    json[] transactionsPayload = [];

    // Gather all transactions to send to analytics service
    stream<record {| string transaction_id; string customer_id; decimal total_amount; string created_at; |}, sql:Error?> allTxStream = dbClient->query(`SELECT transaction_id, customer_id, total_amount, created_at::text FROM transactions ORDER BY created_at ASC`);
    record {| string transaction_id; string customer_id; decimal total_amount; string created_at; |}[] allTransactions = check from var row in allTxStream select row;

    foreach var t in allTransactions {
        // build a minimal json record compatible with analytics-python expectations
        json tx = {
            "transaction_id": t.transaction_id,
            "customer_id": t.customer_id,
            "total_amount": t.total_amount,
            "created_at": t.created_at
        };
        // append to the json array
        transactionsPayload.push(tx);
    }

    boolean analyticsSucceeded = false;
    if (useAnalyticsForSegmentation && allTransactions.length() > 0) {
    int attempts = 0;
        while attempts < analyticsSegmentationMaxAttempts && !analyticsSucceeded {
            attempts += 1;
            // call analytics service
            http:Response|error resp = analyticsClient->post("/customers/segment", transactionsPayload);
            if resp is http:Response {
                if resp.statusCode == 200 {
                    json respJson = check resp.getJsonPayload();
                    if respJson is map<any> {
                        // expected shape: { status: 'success', data: [ { customer_id: ..., Segment: ... }, ... ] }
                        if respJson["data"] is json[] {
                            json[] data = <json[]>respJson["data"];
                            foreach var item in data {
                                if item is map<any> {
                                    any custAny = item["customer_id"];
                                    any segAny = item["Segment"];
                                    string? custId = null;
                                    string? seg = null;
                                    if custAny is string { custId = custAny; }
                                    else if custAny is int { custId = string `${custAny}`; }

                                    if segAny is string { seg = normalizeSegmentName(segAny); }

                                    if custId is string && seg is string {
                                        sql:ExecutionResult res = check dbClient->execute(`UPDATE customers SET segment = ${seg} WHERE customer_id = ${custId}`);
                                        if (res.affectedRowCount > 0) {
                                            _ = check dbClient->execute(`INSERT INTO customer_segments (customer_id, segment) VALUES (${custId}, ${seg})`);
                                            updatedCount += 1;
                                            log:printInfo(string `Updated customer ${custId} to segment: ${seg}`);
                                        }
                                    }
                                }
                            }
                            analyticsSucceeded = true;
                        }
                    }
                } else {
                    string body = check resp.getTextPayload();
                    log:printError("Analytics segmentation returned non-200", keyValues = {"status": string `${resp.statusCode}` , "body": body, "attempt": string `${attempts}`});
                }
            } else {
                string errMsg = "unknown";
                if resp is error { errMsg = resp.message(); }
                log:printError(string `Failed to call analytics service for segmentation (attempt ${attempts}): ${errMsg}`);
            }

            // Backoff disabled: sleep not available in this distribution; retries are immediate.
        }
    }

    if (!analyticsSucceeded) {
        // Fallback: legacy frequency-based segmentation per-customer (safe and deterministic)
        log:printWarn("Analytics segmentation failed or returned no data - falling back to legacy segmentation");
        foreach var customer in customers {
            stream<record {| decimal total_amount; string created_at; |}, sql:Error?> txStream = dbClient->query(`SELECT total_amount, created_at::text FROM transactions WHERE customer_id = ${customer.customer_id} ORDER BY created_at DESC`);
            record {| decimal total_amount; string created_at; |}[] transactions = check from var row in txStream select row;

            string segment = "New";
            if transactions.length() > 0 {
                if transactions.length() >= 5 {
                    segment = "Champion";
                } else if transactions.length() >= 3 {
                    segment = "Loyal";
                } else if transactions.length() >= 1 {
                    // collapse single-purchase customers into "New" for the 4-bucket scheme
                    segment = "New";
                } else {
                    segment = "At-Risk";
                }
            }

            sql:ExecutionResult res = check dbClient->execute(`UPDATE customers SET segment = ${segment} WHERE customer_id = ${customer.customer_id}`);
            if (res.affectedRowCount > 0) {
                _ = check dbClient->execute(`INSERT INTO customer_segments (customer_id, segment) VALUES (${customer.customer_id}, ${segment})`);
                updatedCount += 1;
                log:printInfo(string `Updated customer ${customer.customer_id} to segment (fallback): ${segment}`);
            }
        }
    }

    log:printInfo(string `Customer segmentation complete. Updated ${updatedCount} records.`);
    return { status: "success", message: string `Segmentation complete. Updated ${updatedCount} customers.` };
}

function personalizeMessage(string template, Customer customer, map<string> additionalData) returns string {
    string personalized = "Hi " + customer.name + ", " + template;
    return personalized;
}

// ===== auth/user types =====
type User record {|
    int id;
    string username;
    string role;
    string created_at;
|};

type SignupPayload record {|
    string username;
    string password;
    string role;
|};

type LoginPayload record {|
    string username;
    string password;
|};

type Session record {|
    string token;
    int user_id;
    string expires_at;
|};

// ===== helpers =====
function hashPassword(string password, string salt) returns string {
    byte[] digest = crypto:hashSha256((password + ":" + salt).toBytes());
    string hex = "";
    foreach byte b in digest {
        int ub = <int>b;
        string hx = int:toHexString(ub);
        if ub < 16 { hx = "0" + hx; }
        hex = hex + hx.toLowerAscii();
    }
    return hex;
}

function generateToken() returns string {
    return uuid:createType1AsString();
}

// ===== notification service integration =====
function sendEmailViaSendGrid(string toEmail, string subject, string message, string? photoUrl) returns json|error {
    // mock implementation - log the email instead of sending
    if photoUrl is string {
        log:printInfo("SENDGRID EMAIL WITH PHOTO", keyValues = { 
            "to": toEmail, 
            "subject": subject, 
            "message": message,
            "photo_url": photoUrl,
            "apiKey": sendgrid.apiKey.substring(0, 10) + "..."
        });
    } else {
        log:printInfo("SENDGRID EMAIL", keyValues = { 
            "to": toEmail, 
            "subject": subject, 
            "message": message,
            "apiKey": sendgrid.apiKey.substring(0, 10) + "..."
        });
    }
    
    return { status: "success", message: "Email logged (mock mode)" };
}

// twilio sms integration (mock for now)
function sendSMSViaTwilio(string toPhone, string message) returns json|error {
    log:printInfo("TWILIO SMS", keyValues = { 
        "to": toPhone, 
        "message": message,
        "from": twilio.senderPhoneNumber
    });
    return { status: "success", message: "SMS logged (mock mode)" };
}

function ensureSchema() returns error? {
    // Core domain tables (idempotent)
    _ = check dbClient->execute(`
        CREATE TABLE IF NOT EXISTS products (
            product_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price NUMERIC(10, 2) NOT NULL
        )`);

    _ = check dbClient->execute(`
        CREATE TABLE IF NOT EXISTS customers (
            customer_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            phone_number TEXT UNIQUE,
            segment TEXT DEFAULT 'New'
        )`);

    _ = check dbClient->execute(`
        CREATE TABLE IF NOT EXISTS customer_segments (
            id SERIAL PRIMARY KEY,
            customer_id TEXT NOT NULL REFERENCES customers(customer_id),
            segment TEXT NOT NULL,
            assigned_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )`);

    _ = check dbClient->execute(`
        CREATE TABLE IF NOT EXISTS transactions (
            id SERIAL PRIMARY KEY,
            transaction_id TEXT NOT NULL,
            customer_id TEXT NOT NULL REFERENCES customers(customer_id),
            total_amount NUMERIC(10, 2) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL
        )`);

    _ = check dbClient->execute(`
        CREATE TABLE IF NOT EXISTS transaction_items (
            id SERIAL PRIMARY KEY,
            transaction_record_id INT NOT NULL REFERENCES transactions(id) ON DELETE CASCADE,
            product_id TEXT NOT NULL REFERENCES products(product_id),
            quantity INT NOT NULL,
            price_at_sale NUMERIC(10, 2) NOT NULL
        )`);

    _ = check dbClient->execute(`
        CREATE TABLE IF NOT EXISTS daily_product_sales (
            id SERIAL PRIMARY KEY,
            product_id TEXT NOT NULL REFERENCES products(product_id),
            sale_date DATE NOT NULL,
            total_quantity INT NOT NULL,
            UNIQUE(product_id, sale_date)
        )`);

    _ = check dbClient->execute(`
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(100) UNIQUE NOT NULL,
            password_hash VARCHAR(128) NOT NULL,
            salt VARCHAR(64) NOT NULL,
            role VARCHAR(20) NOT NULL CHECK (role IN ('admin','cashier')),
            created_at TIMESTAMP DEFAULT NOW()
        )`);

    _ = check dbClient->execute(`
        CREATE TABLE IF NOT EXISTS sessions (
            token VARCHAR(64) PRIMARY KEY,
            user_id INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            expires_at TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        )`);

    _ = check dbClient->execute(`
        CREATE TABLE IF NOT EXISTS cashiers (
            id SERIAL PRIMARY KEY,
            user_id INT UNIQUE NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            full_name VARCHAR(150),
            email VARCHAR(150),
            phone VARCHAR(50),
            active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT NOW()
        )`);

            _ = check dbClient->execute(`
        CREATE TABLE IF NOT EXISTS marketing_campaigns (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            message_template TEXT NOT NULL,
            target_segment TEXT NOT NULL,
            campaign_type TEXT DEFAULT 'email',
            email_subject TEXT,
            photo_url TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            created_by TEXT
        )`);
        
        _ = check dbClient->execute(`ALTER TABLE marketing_campaigns ADD COLUMN IF NOT EXISTS campaign_type TEXT DEFAULT 'email'`);
        _ = check dbClient->execute(`ALTER TABLE marketing_campaigns ADD COLUMN IF NOT EXISTS email_subject TEXT`);
        _ = check dbClient->execute(`ALTER TABLE marketing_campaigns ADD COLUMN IF NOT EXISTS photo_url TEXT`);

    _ = check dbClient->execute(`
        CREATE TABLE IF NOT EXISTS notifications (
            id SERIAL PRIMARY KEY,
            campaign_id INT NOT NULL REFERENCES marketing_campaigns(id) ON DELETE CASCADE,
            customer_id TEXT NOT NULL REFERENCES customers(customer_id),
            status TEXT NOT NULL,
            dispatched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            error_message TEXT
        )`);
}

// ===== http service definition =====

@http:ServiceConfig {
    cors: {
        allowOrigins: ["*"],
        allowMethods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allowHeaders: ["Content-Type", "Authorization"]
    }
}
service /api on new http:Listener(9090) {

    function init() returns error? {
        check ensureSchema();
        
        _ = check dbClient->execute(`UPDATE customers SET segment = 'New' WHERE segment IS NULL`);
        log:printInfo("Customer segments initialized");
    }
    
    resource function get health() returns json {
        return { status: "healthy", "service": "Retail Management Ballerina Backend", version: "1.0.0"};
    }
    
    // public endpoint for testing customer data (no auth required)
    resource function get test/customers() returns json|error {
        stream<Customer, sql:Error?> customerStream = dbClient->query(`SELECT customer_id, name, email, phone_number, segment FROM customers ORDER BY name`);
        Customer[] customers = check from var row in customerStream select row;
        return { 
            status: "success", 
            count: customers.length(),
            customers: customers 
        };
    }
    
    // get customer segment history
    resource function get customers/[string customerId]/segments(http:Request req) returns json|error {
        stream<record {| string segment; string assigned_at; |}, sql:Error?> segmentStream = dbClient->query(`SELECT segment, assigned_at::text FROM customer_segments WHERE customer_id = ${customerId} ORDER BY assigned_at DESC`);
        record {| string segment; string assigned_at; |}[] segments = check from var row in segmentStream select row;
        return { 
            status: "success", 
            customer_id: customerId,
            segments: segments 
        };
    }
    
    resource function get products() returns Product[]|error {
        stream<Product, sql:Error?> productStream = dbClient->query(`SELECT product_id, name, category, price FROM products ORDER BY name`);
        return check from var row in productStream select row;
    }

    resource function post products(http:Request req, @http:Payload NewProductPayload payload) returns json|error {
        User|error u = self.requireAuth(req);
        if u is error { return { status: "error", message: "Unauthorized" }; }
        if (<User>u).role != "admin" { return { status: "error", message: "Forbidden" }; }
        _ = check dbClient->execute(`INSERT INTO products (product_id, name, category, price) VALUES (${payload.product_id}, ${payload.name}, ${payload.category}, ${payload.price})`);
        return { status: "success", message: "Product created" };
    }

    resource function put products/[string productId](http:Request req, @http:Payload Product payload) returns json|error {
        User|error u = self.requireAuth(req);
        if u is error { return { status: "error", message: "Unauthorized" }; }
        if (<User>u).role != "admin" { return { status: "error", message: "Forbidden" }; }
        _ = check dbClient->execute(`UPDATE products SET name = ${payload.name}, category = ${payload.category}, price = ${payload.price} WHERE product_id = ${productId}`);
        return { status: "success", message: "Product updated" };
    }

    resource function delete products/[string productId](http:Request req) returns json|error {
        User|error u = self.requireAuth(req);
        if u is error { return { status: "error", message: "Unauthorized" }; }
        if (<User>u).role != "admin" { return { status: "error", message: "Forbidden" }; }
        _ = check dbClient->execute(`DELETE FROM products WHERE product_id = ${productId}`);
        return { status: "success", message: "Product deleted (if existed)" };
    }

    resource function get customers(http:Request req) returns Customer[]|error {
        stream<Customer, sql:Error?> customerStream = dbClient->query(`SELECT customer_id, name, email, phone_number, segment FROM customers ORDER BY name`);
        return check from var row in customerStream select row;
    }

    resource function post customers(@http:Payload NewCustomerPayload payload) returns json|error {
        _ = check dbClient->execute(`INSERT INTO customers (customer_id, name, email, phone_number, segment) VALUES (${payload.customer_id}, ${payload.name}, ${payload.email}, ${payload.phone_number}, 'New')`);
        return { status: "success", message: "Customer created" };
    }



    resource function get transactions() returns Transaction[]|error {
        stream<Transaction, sql:Error?> txStream = dbClient->query(`SELECT transaction_id, customer_id, total_amount, created_at::text FROM transactions ORDER BY created_at DESC LIMIT 100`);
        return check from var row in txStream select row;
    }

    // return latest transactions with optional limit query param for dashboard compatibility
    resource function get transactions/latest(http:Request req) returns Transaction[]|error {
        string? limStr = req.getQueryParamValue("limit");
        int lim = 20;
        if limStr is string {
            int|error parsed = int:fromString(limStr);
            if parsed is int {
                lim = parsed;
            }
        }
        int safe = (lim <= 0 || lim > 200) ? 20 : lim;
        sql:ParameterizedQuery pq = `SELECT transaction_id, customer_id, total_amount, created_at::text FROM transactions ORDER BY created_at DESC LIMIT ${safe}`;
        stream<Transaction, sql:Error?> txStream = dbClient->query(pq);
        return check from var row in txStream select row;
    }

    resource function post transactions(@http:Payload NewTransactionPayload payload) returns json|error {
        transaction {
            record {| int id; |} tx = check dbClient->queryRow(`
                INSERT INTO transactions (transaction_id, customer_id, total_amount, created_at)
                VALUES (${payload.transaction_id}, ${payload.customer_id}, ${payload.total_amount}, NOW())
                RETURNING id
            `);
            int txId = tx.id;
            foreach var item in payload.items {
                _ = check dbClient->execute(`
                    INSERT INTO transaction_items (transaction_record_id, product_id, quantity, price_at_sale)
                    VALUES (${txId}, ${item.product_id}, ${item.quantity}, ${item.price_at_sale})
                `);
                _ = check dbClient->execute(`
                    INSERT INTO daily_product_sales (product_id, sale_date, total_quantity)
                    VALUES (${item.product_id}, CURRENT_DATE, ${item.quantity})
                    ON CONFLICT (product_id, sale_date)
                    DO UPDATE SET total_quantity = daily_product_sales.total_quantity + ${item.quantity}
                `);
            }
            check commit;
        }
        return { status: "success", message: "Transaction created successfully." };
    }

    resource function get sales/data(http:Request req) returns json|error {
        string? productId = req.getQueryParamValue("product_id");
        string? startDate = req.getQueryParamValue("start_date");
        string? endDate = req.getQueryParamValue("end_date");

        sql:ParameterizedQuery pq;
        if productId is string {
            if startDate is string && endDate is string {
                pq = `SELECT ti.product_id, p.name AS product_name, p.category, ti.quantity, ti.price_at_sale, (ti.quantity * ti.price_at_sale) AS line_total, t.transaction_id, t.customer_id, t.created_at FROM transaction_items ti JOIN transactions t ON t.id = ti.transaction_record_id JOIN products p ON p.product_id = ti.product_id WHERE ti.product_id = ${productId} AND t.created_at::date >= ${startDate} AND t.created_at::date <= ${endDate} ORDER BY t.created_at ASC`;
            } else if startDate is string {
                pq = `SELECT ti.product_id, p.name AS product_name, p.category, ti.quantity, ti.price_at_sale, (ti.quantity * ti.price_at_sale) AS line_total, t.transaction_id, t.customer_id, t.created_at FROM transaction_items ti JOIN transactions t ON t.id = ti.transaction_record_id JOIN products p ON p.product_id = ti.product_id WHERE ti.product_id = ${productId} AND t.created_at::date >= ${startDate} ORDER BY t.created_at ASC`;
            } else if endDate is string {
                pq = `SELECT ti.product_id, p.name AS product_name, p.category, ti.quantity, ti.price_at_sale, (ti.quantity * ti.price_at_sale) AS line_total, t.transaction_id, t.customer_id, t.created_at FROM transaction_items ti JOIN transactions t ON t.id = ti.transaction_record_id JOIN products p ON p.product_id = ti.product_id WHERE ti.product_id = ${productId} AND t.created_at::date <= ${endDate} ORDER BY t.created_at ASC`;
            } else {
                pq = `SELECT ti.product_id, p.name AS product_name, p.category, ti.quantity, ti.price_at_sale, (ti.quantity * ti.price_at_sale) AS line_total, t.transaction_id, t.customer_id, t.created_at FROM transaction_items ti JOIN transactions t ON t.id = ti.transaction_record_id JOIN products p ON p.product_id = ti.product_id WHERE ti.product_id = ${productId} ORDER BY t.created_at ASC`;
            }
        } else if startDate is string && endDate is string {
            pq = `SELECT ti.product_id, p.name AS product_name, p.category, ti.quantity, ti.price_at_sale, (ti.quantity * ti.price_at_sale) AS line_total, t.transaction_id, t.customer_id, t.created_at FROM transaction_items ti JOIN transactions t ON t.id = ti.transaction_record_id JOIN products p ON p.product_id = ti.product_id WHERE t.created_at::date >= ${startDate} AND t.created_at::date <= ${endDate} ORDER BY t.created_at ASC`;
        } else if startDate is string {
            pq = `SELECT ti.product_id, p.name AS product_name, p.category, ti.quantity, ti.price_at_sale, (ti.quantity * ti.price_at_sale) AS line_total, t.transaction_id, t.customer_id, t.created_at FROM transaction_items ti JOIN transactions t ON t.id = ti.transaction_record_id JOIN products p ON p.product_id = ti.product_id WHERE t.created_at::date >= ${startDate} ORDER BY t.created_at ASC`;
        } else if endDate is string {
            pq = `SELECT ti.product_id, p.name AS product_name, p.category, ti.quantity, ti.price_at_sale, (ti.quantity * ti.price_at_sale) AS line_total, t.transaction_id, t.customer_id, t.created_at FROM transaction_items ti JOIN transactions t ON t.id = ti.transaction_record_id JOIN products p ON p.product_id = ti.product_id WHERE t.created_at::date <= ${endDate} ORDER BY t.created_at ASC`;
        } else {
            pq = `SELECT ti.product_id, p.name AS product_name, p.category, ti.quantity, ti.price_at_sale, (ti.quantity * ti.price_at_sale) AS line_total, t.transaction_id, t.customer_id, t.created_at FROM transaction_items ti JOIN transactions t ON t.id = ti.transaction_record_id JOIN products p ON p.product_id = ti.product_id ORDER BY t.created_at ASC`;
        }

        stream<record {|
            string product_id;
            string product_name;
            string category;
            int quantity;
            decimal price_at_sale;
            decimal line_total;
            string transaction_id;
            string customer_id;
            string created_at;
        |}, sql:Error?> rows = dbClient->query(pq);

        record {|
            string product_id;
            string product_name;
            string category;
            int quantity;
            decimal price_at_sale;
            decimal line_total;
            string transaction_id;
            string customer_id;
            string created_at;
        |}[] result = check from var r in rows select r;
        return { status: "success", data: result };
    }

    // product sales history used by Analytics for daily forecasts
    resource function get analytics/product/sales_history(http:Request req) returns json|error {
        string? productId = req.getQueryParamValue("product_id");
        if productId is () {
            return { status: "error", message: "product_id is required" };
        }
        string? startDate = req.getQueryParamValue("start_date");
        string? endDate = req.getQueryParamValue("end_date");

        sql:ParameterizedQuery pq;
        if startDate is string && endDate is string {
            pq = `SELECT sale_date::date AS sale_date, total_quantity FROM daily_product_sales WHERE product_id = ${productId} AND sale_date::date >= ${startDate} AND sale_date::date <= ${endDate} ORDER BY sale_date ASC`;
        } else if startDate is string {
            pq = `SELECT sale_date::date AS sale_date, total_quantity FROM daily_product_sales WHERE product_id = ${productId} AND sale_date::date >= ${startDate} ORDER BY sale_date ASC`;
        } else if endDate is string {
            pq = `SELECT sale_date::date AS sale_date, total_quantity FROM daily_product_sales WHERE product_id = ${productId} AND sale_date::date <= ${endDate} ORDER BY sale_date ASC`;
        } else {
            pq = `SELECT sale_date::date AS sale_date, total_quantity FROM daily_product_sales WHERE product_id = ${productId} ORDER BY sale_date ASC`;
        }

        stream<record {| string sale_date; int total_quantity; |}, sql:Error?> rows = dbClient->query(pq);
        record {| string sale_date; int total_quantity; |}[] out = check from var r in rows select r;
        return { status: "success", data: out };
    }

    // seasonal products helper for Analytics/UI
    resource function get products/seasonal() returns json|error {
        stream<Product, sql:Error?> rows = dbClient->query(
            `SELECT product_id, name, category, price
             FROM products
             WHERE LOWER(category) LIKE '%season%'
                OR LOWER(name) LIKE '%christmas%'
                OR LOWER(name) LIKE '%xmas%'
                OR LOWER(name) LIKE '%vesak%'
                OR LOWER(name) LIKE '%awurudu%'
                OR LOWER(name) LIKE '%avurudu%'
             ORDER BY name`
        );
        Product[] products = check from var r in rows select r;
        return { status: "success", data: products };
    }

    // add endpoint for analytics service to get seasonal products
    resource function get analytics/products/seasonal() returns json|error {
        stream<Product, sql:Error?> rows = dbClient->query(
            `SELECT product_id, name, category, price
             FROM products
             WHERE LOWER(category) LIKE '%season%'
                OR LOWER(name) LIKE '%christmas%'
                OR LOWER(name) LIKE '%xmas%'
                OR LOWER(name) LIKE '%vesak%'
                OR LOWER(name) LIKE '%awurudu%'
                OR LOWER(name) LIKE '%avurudu%'
             ORDER BY name`
        );
        Product[] products = check from var r in rows select r;
        return { status: "success", data: products };
    }


    // new endpoint for efficient dashboard metrics
    resource function get analytics/overview() returns json|error {
        // Execute queries sequentially with explicit typedescs for reliability
        record {| decimal total; |} totalRevenueRow = check dbClient->queryRow(
            `SELECT COALESCE(SUM(total_amount), 0) AS total FROM transactions`
        );
        record {| int total; |} transactionCountRow = check dbClient->queryRow(
            `SELECT COUNT(*) AS total FROM transactions`
        );
        record {| int total; |} productCountRow = check dbClient->queryRow(
            `SELECT COUNT(*) AS total FROM products`
        );
        stream<record {| string category; int count; |}, sql:Error?> topCategoriesStream = dbClient->query(
            `SELECT category, COUNT(*) AS count FROM products GROUP BY category ORDER BY count DESC LIMIT 5`
        );
        record {| string category; int count; |}[] topCategories = check from var row in topCategoriesStream select row;

        return {
            totalRevenue: totalRevenueRow.total,
            totalTransactions: transactionCountRow.total,
            totalProducts: productCountRow.total,
            topCategories: topCategories
        };
    }

    resource function post customers/segment(http:Request req) returns json|error {
        
        
        // first, ensure all customers have a default segment
        _ = check dbClient->execute(`UPDATE customers SET segment = 'New' WHERE segment IS NULL`);
        
        return performCustomerSegmentation();
    }

    // add sample data for testing
    resource function post customers/sample(http:Request req) returns json|error {
        
        // add sample customers if none exist
        stream<Customer, sql:Error?> existingCustomers = dbClient->query(`SELECT customer_id FROM customers LIMIT 1`);
        Customer[] customers = check from var row in existingCustomers select row;
        
        if customers.length() == 0 {
            _ = check dbClient->execute(`INSERT INTO customers (customer_id, name, email, phone_number, segment) VALUES 
                ('CUST-001', 'John Doe', 'john@example.com', '+1234567890', 'New'),
                ('CUST-002', 'Jane Smith', 'jane@example.com', '+1234567891', 'Regular'),
                ('CUST-003', 'Bob Johnson', 'bob@example.com', '+1234567892', 'Loyal')`);
            
            return { status: "success", message: "Sample customers added successfully" };
        } else {
            return { status: "success", message: "Customers already exist, no sample data added" };
        }
    }

    // ---- orchestration: proxy inventory predictions to analytics service ----
    resource function post analytics/predict/inventory/[string productId](http:Request req) returns json|error {
        json|error inPayload = req.getJsonPayload();
        map<json> payloadMap = {};
        
        // extract values from incoming payload
        if inPayload is json {
            if inPayload is map<json> {
                payloadMap = <map<json>>inPayload;
            }
        }
        
        // set default values only if not provided
        if !payloadMap.hasKey("days_to_predict") {
            payloadMap["days_to_predict"] = 7;
        }
        if !payloadMap.hasKey("prediction_type") {
            payloadMap["prediction_type"] = "weekly";
        }
        // add product_id to payload for Python service
        payloadMap["product_id"] = productId;
        
        // fetch product details to include in payload
        stream<Product, sql:Error?> productStream = dbClient->query(`SELECT product_id, name, category, price FROM products WHERE product_id = ${productId}`);
        Product[] products = check from var row in productStream select row;
        if products.length() > 0 {
            Product product = products[0];
            payloadMap["product_name"] = product.name;
            payloadMap["category"] = product.category;
        }
        
        // validate and convert values from request
        if payloadMap.hasKey("days_to_predict") {
            if payloadMap["days_to_predict"] is int {
                int days = <int>payloadMap["days_to_predict"];
                if days > 0 && days <= 365 {
                    payloadMap["days_to_predict"] = days;
                }
            } else if payloadMap["days_to_predict"] is string {
                int|error days = int:fromString(<string>payloadMap["days_to_predict"]);
                if days is int && days > 0 && days <= 365 {
                    payloadMap["days_to_predict"] = days;
                }
            }
        }
        if payloadMap.hasKey("prediction_type") && payloadMap["prediction_type"] is string {
            string ptype = <string>payloadMap["prediction_type"];
            if ptype != "" {
                payloadMap["prediction_type"] = ptype;
            }
        }
        
        http:Response resp = check analyticsClient->post(string `/predict/inventory/${productId}`, payloadMap);
        if resp.statusCode != 200 {
            string body = check resp.getTextPayload();
            return error(string `analytics error: ${body}`);
        }
        return check resp.getJsonPayload();
    }

    resource function post analytics/predict/inventory/all(http:Request req) returns json|error {
        json|error may = req.getJsonPayload();
        map<json> payloadMap = {};
        
        // extract values from incoming payload
        if may is json {
            if may is map<json> {
                payloadMap = <map<json>>may;
            }
        }
        
        log:printInfo("Batch prediction request received", keyValues = { "payload": payloadMap.toString() });
        
        // set default values only if not provided
        if !payloadMap.hasKey("days_to_predict") {
            payloadMap["days_to_predict"] = 7;
        }
        
        // validate and convert values from request
        if payloadMap.hasKey("days_to_predict") {
            if payloadMap["days_to_predict"] is int {
                int days = <int>payloadMap["days_to_predict"];
                if days > 0 && days <= 365 {
                    payloadMap["days_to_predict"] = days;
                }
            } else if payloadMap["days_to_predict"] is string {
                int|error days = int:fromString(<string>payloadMap["days_to_predict"]);
                if days is int && days > 0 && days <= 365 {
                    payloadMap["days_to_predict"] = days;
                }
            }
        }
        
        if payloadMap.hasKey("days") {
            if payloadMap["days"] is int {
                int days = <int>payloadMap["days"];
                if days > 0 && days <= 365 {
                    payloadMap["days_to_predict"] = days;
                }
            } else if payloadMap["days"] is string {
                int|error days = int:fromString(<string>payloadMap["days"]);
                if days is int && days > 0 && days <= 365 {
                    payloadMap["days_to_predict"] = days;
                }
            }
        }
        
        http:Response resp = check analyticsClient->post("/predict/inventory/all", payloadMap);
        if resp.statusCode != 200 {
            string body = check resp.getTextPayload();
            return error(string `analytics error: ${body}`);
        }
        return check resp.getJsonPayload();
    }

    // list non-seasonal products directly from database
    resource function get analytics/products/non_seasonal() returns json|error {
        stream<Product, sql:Error?> rows = dbClient->query(
            `SELECT product_id, name, category, price
             FROM products
             WHERE LOWER(category) NOT LIKE '%season%'
                AND LOWER(name) NOT LIKE '%christmas%'
                AND LOWER(name) NOT LIKE '%xmas%'
                AND LOWER(name) NOT LIKE '%vesak%'
                AND LOWER(name) NOT LIKE '%awurudu%'
                AND LOWER(name) NOT LIKE '%avurudu%'
             ORDER BY name`
        );
        Product[] products = check from var r in rows select r;
        log:printInfo("Non-seasonal products fetched", keyValues = { "count": products.length() });
        return { status: "success", data: products };
    }

    resource function get campaigns(http:Request req) returns MarketingCampaign[]|error {
        stream<MarketingCampaign, sql:Error?> campaignStream = dbClient->query(`SELECT id, name, message_template, target_segment, campaign_type, email_subject, photo_url, created_at::text FROM marketing_campaigns ORDER BY created_at DESC`);
        return check from var row in campaignStream select row;
    }

    resource function post campaigns(http:Request req, @http:Payload NewCampaignPayload payload) returns json|error {
        _ = check dbClient->execute(`INSERT INTO marketing_campaigns (name, message_template, target_segment, campaign_type, email_subject, photo_url) VALUES (${payload.name}, ${payload.message_template}, ${payload.target_segment}, ${payload.campaign_type}, ${payload.email_subject}, ${payload.photo_url})`);
        return { status: "success", message: "Campaign created successfully." };
    }
    
    resource function post campaigns/[int campaignId]/launch(http:Request req) returns json|error {
        
        boolean dryRun = false;
        string? dryParam = req.getQueryParamValue("dry_run");
        if dryParam is string {
            string v = dryParam.toLowerAscii();
            if v == "true" || v == "1" || v == "yes" { dryRun = true; }
        }
        
        MarketingCampaign? campaign = check dbClient->queryRow(`SELECT id, name, message_template, target_segment, campaign_type, email_subject, photo_url, created_at::text FROM marketing_campaigns WHERE id = ${campaignId}`);
        if campaign is () {
            return { status: "error", message: "Campaign not found." };
        }

        stream<record {| string customer_id; string name; string? email; string? phone_number; string? segment; |}, sql:Error?> customerStream = dbClient->query(`
            SELECT c.customer_id, c.name, c.email, c.phone_number, c.segment 
            FROM customers c 
            WHERE c.segment = ${campaign.target_segment}
        `);
        record {| string customer_id; string name; string? email; string? phone_number; string? segment; |}[] targetCustomers = check from var row in customerStream select row;

        if targetCustomers.length() == 0 {
            return { status: "success", message: "Campaign launched, but no customers found in the target segment."};
        }

        _ = check dbClient->execute(`UPDATE marketing_campaigns SET created_by = 'admin' WHERE id = ${campaignId}`);

        int sentCount = 0;
        int emailCount = 0;
        int smsCount = 0;
        int failedCount = 0;

        foreach var customer in targetCustomers {
            map<string> additionalData = {
                "expiry_date": "2024-12-31",
                "promo_code": "SAVE20"
            };
            string personalizedMessage = personalizeMessage(campaign.message_template, customer, additionalData);
            string status = "failed";

            if dryRun {
                status = "dry_run";
                sentCount += 1;
                continue;
            }

            if campaign.campaign_type == "email" {
                string? emailOpt = customer.email;
                if emailOpt is string {
                    string email = emailOpt;
                    string emailSubject = campaign.email_subject ?: "Special Offer from Your Store";
                    json|error emailResult = sendEmailViaSendGrid(email, emailSubject, personalizedMessage, campaign.photo_url);
                    if emailResult is json {
                        status = "sent_email";
                        sentCount += 1;
                        emailCount += 1;
                        log:printInfo(string `Email sent successfully to ${email}`);
                    } else {
                        log:printError(string `Failed to send email to ${email}: ${emailResult.toString()}`);
                        status = "failed_email";
                        failedCount += 1;
                    }
                } else {
                    log:printWarn(string `Email campaign but no email for customer ${customer.customer_id}`);
                    status = "no_email";
                    failedCount += 1;
                }
            } else if campaign.campaign_type == "sms" {
                string? phoneOpt = customer.phone_number;
                if phoneOpt is string {
                    string phone = phoneOpt;
                    json|error smsResult = sendSMSViaTwilio(phone, personalizedMessage);
                    if smsResult is json {
                        status = "sent_sms";
                        sentCount += 1;
                        smsCount += 1;
                        log:printInfo(string `SMS sent successfully to ${phone}`);
                    } else {
                        log:printError(string `Failed to send SMS to ${phone}: ${smsResult.toString()}`);
                        status = "failed_sms";
                        failedCount += 1;
                    }
                } else {
                    log:printWarn(string `SMS campaign but no phone for customer ${customer.customer_id}`);
                    status = "no_phone";
                    failedCount += 1;
                }
            } else {
                string? emailOpt = customer.email;
                if emailOpt is string {
                    string email = emailOpt;
                    string emailSubject = campaign.email_subject ?: "Special Offer from Your Store";
                    json|error emailResult = sendEmailViaSendGrid(email, emailSubject, personalizedMessage, campaign.photo_url);
                    if emailResult is json {
                        status = "sent_email";
                        sentCount += 1;
                        emailCount += 1;
                        log:printInfo(string `Email sent successfully to ${email}`);
                    } else {
                        log:printError(string `Failed to send email to ${email}: ${emailResult.toString()}`);
                        status = "failed_email";
                        failedCount += 1;
                    }
                } else {
                    string? phoneOpt = customer.phone_number;
                    if phoneOpt is string {
                        string phone = phoneOpt;
                        json|error smsResult = sendSMSViaTwilio(phone, personalizedMessage);
                        if smsResult is json {
                            status = "sent_sms";
                            sentCount += 1;
                            smsCount += 1;
                            log:printInfo(string `SMS sent successfully to ${phone}`);
                        } else {
                            log:printError(string `Failed to send SMS to ${phone}: ${smsResult.toString()}`);
                            status = "failed_sms";
                            failedCount += 1;
                        }
                    } else {
                        log:printWarn(string `No contact method available for customer ${customer.customer_id}`);
                        status = "no_contact_method";
                        failedCount += 1;
                    }
                }
            }

            _ = check dbClient->execute(`
                INSERT INTO notifications (campaign_id, customer_id, status)
                VALUES (${campaignId}, ${customer.customer_id}, ${status})
            `);
        }

        log:printInfo(string `Campaign ${campaignId} completed. Successful: ${sentCount}, Failed: ${failedCount}`);

        string mode = dryRun ? "(dry-run) " : "";
        log:printInfo(string `Campaign ${campaignId} launched ${mode}Attempted to notify ${targetCustomers.length()} customers. Successful: ${sentCount} (${emailCount} emails, ${smsCount} SMS), Failed: ${failedCount}`);
        
        if dryRun {
            return { status: "success", message: string `Dry-run: ${sentCount} customers would be notified.` };
        }
        return { status: "success", message: string `Campaign launched. ${sentCount} notifications dispatched (${emailCount} emails, ${smsCount} SMS). ${failedCount} failed.` };
    }

    // ===== auth & user management =====
    resource function post auth/signup(@http:Payload SignupPayload payload) returns json|error {
        string role = payload.role.toLowerAscii();
        if role != "admin" && role != "cashier" {
            return { status: "error", message: "Invalid role. Use 'admin' or 'cashier'." };
        }
        string salt = uuid:createType4AsString();
        string pwdHash = hashPassword(payload.password, salt);
        // Insert user
        sql:ParameterizedQuery pq = `INSERT INTO users (username, password_hash, salt, role) VALUES (${payload.username}, ${pwdHash}, ${salt}, ${role}) RETURNING id`;
    record {| int id; |} row = check dbClient->queryRow(pq);
    int userId = row.id;
        if role == "cashier" {
            _ = check dbClient->execute(`INSERT INTO cashiers (user_id) VALUES (${userId})`);
        }
        return { status: "success", message: "Signup successful", user_id: userId, role: role };
    }

    resource function post auth/login(@http:Payload LoginPayload payload) returns json|error {
        record {| int id; string password_hash; string salt; string role; |}|() row = check dbClient->queryRow(`SELECT id, password_hash, salt, role FROM users WHERE username = ${payload.username}`);
        if row is () { return { status: "error", message: "Invalid credentials" }; }
        string calc = hashPassword(payload.password, row.salt);
        if calc != row.password_hash { return { status: "error", message: "Invalid credentials" }; }
    string token = generateToken();
    _ = check dbClient->execute(`INSERT INTO sessions (token, user_id, expires_at) VALUES (${token}, ${row.id}, NOW() + INTERVAL '12 HOURS')`);
        return { status: "success", token: token, role: row.role };
    }

    resource function post auth/logout(http:Request req) returns json|error {
    string|http:HeaderNotFoundError hdr = req.getHeader("authorization");
    if hdr is http:HeaderNotFoundError { return { status: "success" }; }
    string token = hdr.toLowerAscii().startsWith("bearer ") ? hdr.substring(7) : hdr;
        _ = check dbClient->execute(`DELETE FROM sessions WHERE token = ${token}`);
        return { status: "success", message: "Logged out" };
    }

    resource function get me(http:Request req) returns json|error {
    User|error u = self.requireAuth(req);
        if u is error { return { status: "error", message: u.toString() }; }
        return { status: "success", user: u };
    }

    resource function get cashiers(http:Request req) returns json|error {
    User|error u = self.requireAuth(req);
        if u is error { return { status: "error", message: "Unauthorized" }; }
        if (<User>u).role != "admin" { return { status: "error", message: "Forbidden" }; }
        stream<record {| int id; int user_id; string? full_name; string? email; string? phone; boolean active; string created_at; string username; |}, sql:Error?> rows = dbClient->query(`
            SELECT c.id, c.user_id, c.full_name, c.email, c.phone, c.active, c.created_at::text, u.username
            FROM cashiers c JOIN users u ON u.id = c.user_id ORDER BY c.created_at DESC`);
        record {| int id; int user_id; string? full_name; string? email; string? phone; boolean active; string created_at; string username; |}[] list = check from var r in rows select r;
        return { status: "success", data: list };
    }

    resource function post cashiers(http:Request req, @http:Payload record {| int user_id; string? full_name; string? email; string? phone; |} payload) returns json|error {
    User|error u = self.requireAuth(req);
        if u is error { return { status: "error", message: "Unauthorized" }; }
        if (<User>u).role != "admin" { return { status: "error", message: "Forbidden" }; }
        _ = check dbClient->execute(`INSERT INTO cashiers (user_id, full_name, email, phone) VALUES (${payload.user_id}, ${payload.full_name}, ${payload.email}, ${payload.phone}) ON CONFLICT (user_id) DO UPDATE SET full_name = EXCLUDED.full_name, email = EXCLUDED.email, phone = EXCLUDED.phone`);
        return { status: "success", message: "Cashier profile upserted" };
    }

    // ===== search endpoints for pos =====
    resource function get search/products(http:Request req) returns json|error {
    string q = req.getQueryParamValue("q") ?: "";
    string like = "%" + q.toLowerAscii() + "%";
        stream<Product, sql:Error?> rows = dbClient->query(`
            SELECT product_id, name, category, price FROM products
            WHERE LOWER(name) LIKE ${like} OR LOWER(product_id) LIKE ${like} OR LOWER(category) LIKE ${like}
            ORDER BY name LIMIT 50`);
        Product[] out = check from var r in rows select r;
        return { status: "success", data: out };
    }

    resource function get search/customers(http:Request req) returns json|error {
    string q = req.getQueryParamValue("q") ?: "";
    string like = "%" + q.toLowerAscii() + "%";
        stream<Customer, sql:Error?> rows = dbClient->query(`
            SELECT customer_id, name, email, phone_number, segment FROM customers
            WHERE LOWER(name) LIKE ${like} OR LOWER(customer_id) LIKE ${like} OR LOWER(email) LIKE ${like} OR LOWER(phone_number) LIKE ${like}
            ORDER BY name LIMIT 50`);
        Customer[] out = check from var r in rows select r;
        return { status: "success", data: out };
    }

    // helper: auth check
    function requireAuth(http:Request req) returns User|error {
        string|http:HeaderNotFoundError hdr = req.getHeader("authorization");
        if hdr is http:HeaderNotFoundError { return error("Missing Authorization header"); }
        string token = hdr.toLowerAscii().startsWith("bearer ") ? hdr.substring(7) : hdr;
        record {| int id; string username; string role; string created_at; string expires_at; |}|() row = check dbClient->queryRow(`
            SELECT u.id, u.username, u.role, u.created_at::text, s.expires_at::text
            FROM sessions s JOIN users u ON u.id = s.user_id
            WHERE s.token = ${token} AND s.expires_at > NOW()`);
        if row is () { return error("Invalid token"); }
        return { id: row.id, username: row.username, role: row.role, created_at: row.created_at };
    }
}
