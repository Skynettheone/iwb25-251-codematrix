
BALLERINA_BACKEND_URL = "http://localhost:9090"
ANALYTICS_API_PORT = 8000

BALLERINA_ENDPOINTS = {
    'products': '/api/products',
    'sales_data': '/api/sales/data',
    'daily_sales': '/api/sales/daily-summary',
    'transactions': '/api/transactions',
    'customers': '/api/customers'
}

ML_CONFIG = {
    'min_data_points': 10,
    'default_prediction_days': 30,
    'confidence_threshold': 0.6,
    'seasonality_threshold': 0.3
}

CHART_CONFIG = {
    'figure_size': (12, 6),
    'dpi': 100,
    'chart_colors': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
}
