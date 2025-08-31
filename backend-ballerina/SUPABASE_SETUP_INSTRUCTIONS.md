# Supabase Database Setup Instructions

## When Connection Fails

If you get: `Error in SQL connector configuration: Failed to initialize pool: The connection attempt failed. Caused by :db.jknffzcmoojysdecocbj.supabase.co`

**Causes:**
- Supabase service offline
- Network/firewall blocking
- Expired credentials
- IP restrictions

## Solution: Setup New Supabase

### Step 1: Create New Project
1. Go to [supabase.com](https://supabase.com)
2. Create new project
3. Wait 2-3 minutes for setup

### Step 2: Get Credentials
- Go to Project Settings → Database
- Copy: Host, Database, Port, Username, Password

### Step 3: Update Config.toml
```toml
host = "your-new-db-host.supabase.co"
username = "postgres"
password = "your-new-password"
database = "postgres"
port = 5432
```

### Step 4: Create Tables (Run in Order)

**1. Base Tables:**
```sql
CREATE TABLE products (
    product_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    price NUMERIC(10, 2) NOT NULL
);

CREATE TABLE customers (
    customer_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE,
    phone_number TEXT UNIQUE,
    segment TEXT DEFAULT 'New'
);

CREATE TABLE customer_segments (
    id SERIAL PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    segment TEXT NOT NULL,
    assigned_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE transactions (
    id SERIAL PRIMARY KEY,
    transaction_id TEXT NOT NULL,
    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    total_amount NUMERIC(10, 2) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE transaction_items (
    id SERIAL PRIMARY KEY,
    transaction_record_id INT NOT NULL REFERENCES transactions(id) ON DELETE CASCADE,
    product_id TEXT NOT NULL REFERENCES products(product_id),
    quantity INT NOT NULL,
    price_at_sale NUMERIC(10, 2) NOT NULL
);

CREATE TABLE marketing_campaigns (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    message_template TEXT NOT NULL,
    target_segment TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by TEXT
);

CREATE TABLE notifications (
    id SERIAL PRIMARY KEY,
    campaign_id INT NOT NULL REFERENCES marketing_campaigns(id) ON DELETE CASCADE,
    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    status TEXT NOT NULL,
    dispatched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    error_message TEXT
);

CREATE TABLE daily_product_sales (
    id SERIAL PRIMARY KEY,
    product_id TEXT NOT NULL REFERENCES products(product_id),
    sale_date DATE NOT NULL,
    total_quantity INT NOT NULL,
    UNIQUE(product_id, sale_date)
);

CREATE TABLE marketing_campaigns (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    message_template TEXT NOT NULL,
    target_segment TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by TEXT
);

CREATE TABLE notifications (
    id SERIAL PRIMARY KEY,
    campaign_id INT NOT NULL REFERENCES marketing_campaigns(id) ON DELETE CASCADE,
    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    status TEXT NOT NULL,
    dispatched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    error_message TEXT
);
```

**2. Insert Data:**

**Customers (Initial):**
```sql
INSERT INTO customers (customer_id, name, segment) VALUES
('CUST-101', 'Nimal Perera', 'Champion'),
('CUST-102', 'Sunil Silva', 'Loyal'),
('CUST-103', 'Kamala Fernando', 'Loyal'),
('CUST-104', 'Ravi Jayasuriya', 'New'),
('CUST-105', 'Anusha Bandara', 'At-Risk');

INSERT INTO customers (customer_id, name, email, phone_number, segment) VALUES
('CUST-999', 'Demo User', 'tharukakarunanayaka7@gmail.com', '+94 72 883 0581', 'Champion');

-- Update customer details with email and phone
UPDATE customers
SET email = 'nimal.perera@example.com', phone_number = '+94 77 123 4567'
WHERE customer_id = 'CUST-101';

UPDATE customers
SET email = 'sunil.silva@example.com', phone_number = '+94 71 234 5678'
WHERE customer_id = 'CUST-102';

UPDATE customers
SET email = 'kamala.fernando@example.com', phone_number = '+94 76 345 6789'
WHERE customer_id = 'CUST-103';

UPDATE customers
SET email = 'ravi.jayasuriya@example.com', phone_number = '+94 75 456 7890'
WHERE customer_id = 'CUST-104';

UPDATE customers
SET email = 'anusha.bandara@example.com', phone_number = '+94 70 567 8901'
WHERE customer_id = 'CUST-105';
```

-- Products
```sql
INSERT INTO products (product_id, name, category, price) VALUES
('PROD-BREAD-L', 'Roast Paan', 'Bakery', 150.00),
('PROD-DHAL-1KG', 'Dhal (Lentils) 1kg', 'Groceries', 250.00),
('PROD-RICE-5KG', 'Keeri Samba Rice 5kg', 'Groceries', 500.00),
('PROD-COCONUT', 'Coconut', 'Groceries', 80.00),
('PROD-SAMBOL-P', 'Pol Sambol Packet', 'Ready Meals', 100.00),
('PROD-TEA-100G', 'Tea Leaves 100g', 'Beverages', 200.00),
('PROD-MILK-PDR-400G', 'Milk Powder 400g', 'Groceries', 450.00),
('PROD-KOTTU-CKN', 'Chicken Kottu', 'Ready Meals', 400.00),
('PROD-EGGS', 'Eggs (Dozen)', 'Groceries', 600.00),
('SEASONAL-AVURUDU-KAVUM', 'Kavum (Packet of 10)', 'Seasonal', 350.00),
('SEASONAL-AVURUDU-KOKIS', 'Kokis (Packet of 20)', 'Seasonal', 300.00),
('SEASONAL-AVURUDU-ASMI', 'Asmi (Packet)', 'Seasonal', 400.00),
('SEASONAL-VESAK-LANTERN', 'Vesak Lantern (Medium)', 'Seasonal', 250.00),
('SEASONAL-VESAK-BUCKET', 'Vesak Bucket', 'Seasonal', 150.00),
('SEASONAL-VESAK-LAMPS', 'Clay Oil Lamps (Packet of 10)', 'Seasonal', 200.00),
('SEASONAL-XMAS-CAKE-500G', 'Christmas Cake (500g)', 'Seasonal', 1200.00),
('SEASONAL-XMAS-BREUDHER', 'Breudher (Small)', 'Seasonal', 800.00),
('SEASONAL-XMAS-DECO', 'Christmas Decorations (Set)', 'Seasonal', 1500.00);
```

-- Historical Transactions (2024-2025)
```sql
INSERT INTO transactions (id, transaction_id, customer_id, total_amount, created_at) VALUES
(1, 'TXN-20240105-001', 'CUST-101', 3500, '2024-01-05 11:00:00+05:30'),
(2, 'TXN-20240106-001', 'CUST-102', 4160, '2024-01-06 19:00:00+05:30'),
(3, 'TXN-20240210-001', 'CUST-103', 800, '2024-02-10 12:00:00+05:30'),
(4, 'TXN-20240211-001', 'CUST-104', 5000, '2024-02-11 17:00:00+05:30'),
(5, 'TXN-20240320-001', 'CUST-101', 1350, '2024-03-20 09:00:00+05:30'),
(10, 'TXN-20240408-001', 'CUST-105', 13000, '2024-04-08 10:00:00+05:30'),
(11, 'TXN-20240409-001', 'CUST-101', 19500, '2024-04-09 11:30:00+05:30'),
(12, 'TXN-20240410-001', 'CUST-102', 20500, '2024-04-10 15:00:00+05:30'),
(13, 'TXN-20240411-001', 'CUST-103', 10250, '2024-04-11 18:00:00+05:30'),
(20, 'TXN-20240522-001', 'CUST-104', 7500, '2024-05-22 19:00:00+05:30'),
(21, 'TXN-20240523-001', 'CUST-105', 4750, '2024-05-23 12:00:00+05:30'),
(25, 'TXN-20240615-001', 'CUST-101', 4500, '2024-06-15 14:00:00+05:30'),
(26, 'TXN-20240720-001', 'CUST-102', 1200, '2024-07-20 11:00:00+05:30'),
(27, 'TXN-20240901-001', 'CUST-103', 5000, '2024-09-01 10:00:00+05:30'),
(28, 'TXN-20241025-001', 'CUST-104', 1800, '2024-10-25 16:00:00+05:30'),
(30, 'TXN-20241220-001', 'CUST-105', 22500, '2024-12-20 16:00:00+05:30'),
(31, 'TXN-20241222-001', 'CUST-101', 13600, '2024-12-22 17:00:00+05:30'),
(32, 'TXN-20241223-001', 'CUST-102', 17900, '2024-12-23 11:00:00+05:30'),
(100, 'TXN-20250115-001', 'CUST-103', 5000, '2025-01-15 12:00:00+05:30'),
(101, 'TXN-20250210-001', 'CUST-104', 3200, '2025-02-10 18:30:00+05:30'),
(102, 'TXN-20250305-001', 'CUST-101', 750, '2025-03-05 09:00:00+05:30'),
(110, 'TXN-20250409-001', 'CUST-102', 15600, '2025-04-09 13:00:00+05:30'),
(111, 'TXN-20250410-001', 'CUST-105', 7800, '2025-04-10 15:00:00+05:30'),
(120, 'TXN-20250518-001', 'CUST-103', 6000, '2025-05-18 20:00:00+05:30'),
(121, 'TXN-20250519-001', 'CUST-104', 3000, '2025-05-19 11:00:00+05:30'),
(130, 'TXN-20250725-001', 'CUST-101', 10000, '2025-07-25 10:00:00+05:30'),
(131, 'TXN-20250810-001', 'CUST-102', 2400, '2025-08-10 19:00:00+05:30');
```

-- Recent Transactions (Last 3 months)
```sql
INSERT INTO transactions (transaction_id, customer_id, total_amount, created_at) VALUES
('TXN-001', 'CUST-101', 400.00, NOW() - INTERVAL '88 days'),
('TXN-002', 'CUST-102', 680.00, NOW() - INTERVAL '87 days'),
('TXN-003', 'CUST-103', 800.00, NOW() - INTERVAL '86 days'),
('TXN-004', 'CUST-101', 1050.00, NOW() - INTERVAL '81 days'),
('TXN-005', 'CUST-104', 160.00, NOW() - INTERVAL '80 days'),
('TXN-006', 'CUST-102', 750.00, NOW() - INTERVAL '79 days'),
('TXN-007', 'CUST-105', 1200.00, NOW() - INTERVAL '75 days'),
('TXN-008', 'CUST-101', 150.00, NOW() - INTERVAL '74 days'),
('TXN-009', 'CUST-103', 500.00, NOW() - INTERVAL '72 days'),
('TXN-010', 'CUST-102', 80.00, NOW() - INTERVAL '68 days'),
('TXN-011', 'CUST-104', 1000.00, NOW() - INTERVAL '65 days'),
('TXN-012', 'CUST-101', 400.00, NOW() - INTERVAL '61 days'),
('TXN-013', 'CUST-105', 250.00, NOW() - INTERVAL '58 days'),
('TXN-014', 'CUST-102', 800.00, NOW() - INTERVAL '55 days'),
('TXN-015', 'CUST-103', 650.00, NOW() - INTERVAL '51 days'),
('TXN-016', 'CUST-101', 160.00, NOW() - INTERVAL '48 days'),
('TXN-017', 'CUST-104', 500.00, NOW() - INTERVAL '45 days'),
('TXN-018', 'CUST-102', 1100.00, NOW() - INTERVAL '42 days'),
('TXN-019', 'CUST-105', 400.00, NOW() - INTERVAL '38 days'),
('TXN-020', 'CUST-101', 80.00, NOW() - INTERVAL '35 days'),
('TXN-021', 'CUST-103', 1500.00, NOW() - INTERVAL '31 days'),
('TXN-022', 'CUST-102', 250.00, NOW() - INTERVAL '28 days'),
('TXN-023', 'CUST-104', 400.00, NOW() - INTERVAL '24 days'),
('TXN-024', 'CUST-101', 800.00, NOW() - INTERVAL '20 days'),
('TXN-025', 'CUST-105', 150.00, NOW() - INTERVAL '16 days'),
('TXN-026', 'CUST-102', 1000.00, NOW() - INTERVAL '12 days'),
('TXN-027', 'CUST-103', 680.00, NOW() - INTERVAL '8 days'),
('TXN-028', 'CUST-101', 80.00, NOW() - INTERVAL '5 days'),
('TXN-029', 'CUST-104', 1050.00, NOW() - INTERVAL '3 days'),
('TXN-030', 'CUST-102', 400.00, NOW() - INTERVAL '1 day');
```

-- Transaction Items for Historical Transactions
```sql
INSERT INTO transaction_items (transaction_record_id, product_id, quantity, price_at_sale) VALUES
(1, 'PROD-RICE-5KG', 7, 500), (2, 'PROD-RICE-5KG', 8, 500), (2, 'PROD-COCONUT', 2, 80),
(3, 'PROD-KOTTU-CKN', 2, 400), (4, 'PROD-RICE-5KG', 10, 500), (5, 'PROD-MILK-PDR-400G', 3, 450),
(10, 'SEASONAL-AVURUDU-KAVUM', 20, 350), (10, 'SEASONAL-AVURUDU-KOKIS', 20, 300),
(11, 'SEASONAL-AVURUDU-KAVUM', 30, 350), (11, 'SEASONAL-AVURUDU-KOKIS', 30, 300),
(12, 'SEASONAL-AVURUDU-KAVUM', 35, 350), (12, 'SEASONAL-AVURUDU-KOKIS', 25, 300), (12, 'SEASONAL-AVURUDU-ASMI', 2, 400),
(13, 'SEASONAL-AVURUDU-KAVUM', 15, 350), (13, 'SEASONAL-AVURUDU-KOKIS', 15, 300), (13, 'SEASONAL-AVURUDU-ASMI', 2, 400),
(20, 'SEASONAL-VESAK-LANTERN', 30, 250),
(21, 'SEASONAL-VESAK-LANTERN', 15, 250), (21, 'SEASONAL-VESAK-BUCKET', 5, 150), (21, 'SEASONAL-VESAK-LAMPS', 2, 200),
(25, 'PROD-RICE-5KG', 9, 500), (26, 'PROD-EGGS', 2, 600), (27, 'PROD-RICE-5KG', 10, 500), (28, 'PROD-EGGS', 3, 600),
(30, 'SEASONAL-XMAS-CAKE-500G', 15, 1200), (30, 'SEASONAL-XMAS-DECO', 3, 1500),
(31, 'SEASONAL-XMAS-CAKE-500G', 8, 1200), (31, 'SEASONAL-XMAS-BREUDHER', 5, 800),
(32, 'SEASONAL-XMAS-CAKE-500G', 12, 1200), (32, 'SEASONAL-XMAS-DECO', 1, 1500), (32, 'SEASONAL-XMAS-BREUDHER', 2, 800),
(100, 'PROD-RICE-5KG', 10, 500.00), (101, 'PROD-KOTTU-CKN', 8, 400.00), (102, 'PROD-DHAL-1KG', 3, 250.00),
(110, 'SEASONAL-AVURUDU-KAVUM', 24, 350.00), (110, 'SEASONAL-AVURUDU-KOKIS', 24, 300.00),
(111, 'SEASONAL-AVURUDU-KAVUM', 12, 350.00), (111, 'SEASONAL-AVURUDU-KOKIS', 12, 300.00),
(120, 'SEASONAL-VESAK-LANTERN', 24, 250.00),
(121, 'SEASONAL-VESAK-LANTERN', 12, 250.00),
(130, 'PROD-RICE-5KG', 20, 500.00),
(131, 'PROD-EGGS', 4, 600.00);
```

-- Transaction Items for Recent Transactions
```sql
INSERT INTO transaction_items (transaction_record_id, product_id, quantity, price_at_sale) VALUES
(1, 'PROD-BREAD-L', 1, 150.00), (1, 'PROD-DHAL-1KG', 1, 250.00),
(2, 'PROD-RICE-5KG', 1, 500.00), (2, 'PROD-COCONUT', 1, 80.00), (2, 'PROD-SAMBOL-P', 1, 100.00),
(3, 'PROD-KOTTU-CKN', 2, 400.00),
(4, 'PROD-RICE-5KG', 1, 500.00), (4, 'PROD-MILK-PDR-400G', 1, 450.00), (4, 'PROD-SAMBOL-P', 1, 100.00),
(5, 'PROD-COCONUT', 2, 80.00),
(6, 'PROD-RICE-5KG', 1, 500.00), (6, 'PROD-DHAL-1KG', 1, 250.00),
(7, 'PROD-EGGS', 2, 600.00),
(8, 'PROD-BREAD-L', 1, 150.00),
(9, 'PROD-RICE-5KG', 1, 500.00),
(10, 'PROD-COCONUT', 1, 80.00),
(11, 'PROD-RICE-5KG', 2, 500.00),
(12, 'PROD-KOTTU-CKN', 1, 400.00),
(13, 'PROD-DHAL-1KG', 1, 250.00),
(14, 'PROD-KOTTU-CKN', 2, 400.00),
(15, 'PROD-TEA-100G', 1, 200.00), (15, 'PROD-MILK-PDR-400G', 1, 450.00),
(16, 'PROD-COCONUT', 2, 80.00),
(17, 'PROD-RICE-5KG', 1, 500.00),
(18, 'PROD-RICE-5KG', 1, 500.00), (18, 'PROD-EGGS', 1, 600.00),
(19, 'PROD-KOTTU-CKN', 1, 400.00),
(20, 'PROD-COCONUT', 1, 80.00),
(21, 'PROD-RICE-5KG', 3, 500.00),
(22, 'PROD-DHAL-1KG', 1, 250.00),
(23, 'PROD-BREAD-L', 1, 150.00), (23, 'PROD-SAMBOL-P', 1, 100.00), (23, 'PROD-COCONUT', 2, 80.00),
(24, 'PROD-KOTTU-CKN', 2, 400.00),
(25, 'PROD-BREAD-L', 1, 150.00),
(26, 'PROD-RICE-5KG', 2, 500.00),
(27, 'PROD-RICE-5KG', 1, 500.00), (27, 'PROD-COCONUT', 1, 80.00), (27, 'PROD-SAMBOL-P', 1, 100.00),
(28, 'PROD-COCONUT', 1, 80.00),
(29, 'PROD-RICE-5KG', 1, 500.00), (29, 'PROD-MILK-PDR-400G', 1, 450.00), (29, 'PROD-SAMBOL-P', 1, 100.00),
(30, 'PROD-KOTTU-CKN', 1, 400.00);
```

**3. Functions & Triggers (Run Last):**
```sql
CREATE OR REPLACE FUNCTION update_daily_sales()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO daily_product_sales (product_id, sale_date, total_quantity)
    VALUES (NEW.product_id, CURRENT_DATE, NEW.quantity)
    ON CONFLICT (product_id, sale_date)
    DO UPDATE SET total_quantity = daily_product_sales.total_quantity + NEW.quantity;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER on_new_transaction_item
AFTER INSERT ON transaction_items
FOR EACH ROW
EXECUTE FUNCTION update_daily_sales();
```

### Step 5: Test & Verify
1. Restart Ballerina app
2. Test connection
3. Verify data with: `SELECT COUNT(*) FROM customers;`

## Note
- Run queries in order: Tables → Data → Functions
- Backup existing data first
- Check IP restrictions in Supabase
