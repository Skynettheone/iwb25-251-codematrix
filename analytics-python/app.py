# app.py - v5 Enhanced Analytics Service

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
import uvicorn
from datetime import datetime, timedelta

# --- Pydantic Models for Data Validation ---

class DailySaleInfo(BaseModel):
    sale_date: str
    total_quantity: int

class PredictionRequest(BaseModel):
    product_id: str
    history: List[DailySaleInfo]
    days_to_predict: int = 7
    prediction_type: str = "weekly"  # "weekly" or "monthly"

class StockPrediction(BaseModel):
    product_id: str
    forecast_period_days: int
    prediction_type: str
    total_forecast: int
    daily_forecast: List[int]
    weekly_averages: Optional[List[float]] = None  # For monthly predictions
    monthly_averages: Optional[List[float]] = None  # For yearly view
    confidence_score: float
    trend_direction: str  # "increasing", "decreasing", "stable"
    message: str

class AggregatedSalesData(BaseModel):
    period: str  # Week number or month name
    average_sales: float
    total_sales: int
    period_type: str  # "week" or "month"

# --- FastAPI Application ---
app = FastAPI(title="Enhanced Analytics Service", description="Advanced stock prediction and sales analytics")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:9090"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper Functions ---

def calculate_trend_direction(daily_forecast):
    """Determine if sales trend is increasing, decreasing, or stable"""
    if len(daily_forecast) < 3:
        return "stable"
    
    first_third = np.mean(daily_forecast[:len(daily_forecast)//3])
    last_third = np.mean(daily_forecast[-len(daily_forecast)//3:])
    
    change_percentage = ((last_third - first_third) / first_third) * 100 if first_third > 0 else 0
    
    if change_percentage > 10:
        return "increasing"
    elif change_percentage < -10:
        return "decreasing"
    else:
        return "stable"

def calculate_confidence_score(model, X, y):
    """Calculate prediction confidence based on model performance"""
    try:
        score = model.score(X, y)
        return max(0.0, min(1.0, score))
    except:
        return 0.7  # Default confidence

def get_aggregated_sales_data(df, prediction_type):
    """Calculate weekly/monthly averages for visualization"""
    df['sale_date'] = pd.to_datetime(df['sale_date'])
    
    if prediction_type == "weekly":
        # Group by week for monthly prediction view
        df['week'] = df['sale_date'].dt.isocalendar().week
        weekly_data = df.groupby('week')['total_quantity'].agg(['mean', 'sum']).reset_index()
        return weekly_data['mean'].tolist(), weekly_data['sum'].tolist()
    else:
        # Group by month for yearly prediction view
        df['month'] = df['sale_date'].dt.month
        monthly_data = df.groupby('month')['total_quantity'].agg(['mean', 'sum']).reset_index()
        return monthly_data['mean'].tolist(), monthly_data['sum'].tolist()


# --- API Endpoints ---

@app.get("/status")
def get_status():
    return {"status": "Enhanced Analytics service running", "version": "2.0.0"}

@app.post("/predict/stock", response_model=StockPrediction)
async def predict_stock_enhanced(request: PredictionRequest):
    if len(request.history) < 3:
        raise HTTPException(
            status_code=400, 
            detail="Insufficient historical data. At least 3 daily records required for enhanced predictions."
        )

    # Prepare data
    df = pd.DataFrame([item.dict() for item in request.history])
    df['sale_date'] = pd.to_datetime(df['sale_date'])
    df = df.sort_values(by='sale_date')
    
    # Feature engineering
    df['day_number'] = (df['sale_date'] - df['sale_date'].min()).dt.days
    df['day_of_week'] = df['sale_date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Use Random Forest for better predictions
    X = df[['day_number', 'day_of_week', 'is_weekend']]
    y = df['total_quantity']
    
    # Scale features for better performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Use Random Forest for more robust predictions
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_scaled, y)
    
    # Calculate confidence score
    confidence = calculate_confidence_score(model, X_scaled, y)
    
    # Prepare future prediction features
    last_day = df['day_number'].max()
    future_features = []
    
    for i in range(1, request.days_to_predict + 1):
        future_day = last_day + i
        future_date = df['sale_date'].max() + timedelta(days=i)
        day_of_week = future_date.dayofweek
        is_weekend = 1 if day_of_week >= 5 else 0
        
        future_features.append([future_day, day_of_week, is_weekend])
    
    future_features_scaled = scaler.transform(future_features)
    predicted_quantities = model.predict(future_features_scaled)
    
    # Ensure non-negative predictions
    daily_forecast = [max(0, int(q)) for q in predicted_quantities]
    total_forecast = sum(daily_forecast)
    
    # Calculate aggregated data for visualization
    weekly_avg, monthly_avg = get_aggregated_sales_data(df, request.prediction_type)
    
    # Determine trend direction
    trend = calculate_trend_direction(daily_forecast)
    
    return StockPrediction(
        product_id=request.product_id,
        forecast_period_days=request.days_to_predict,
        prediction_type=request.prediction_type,
        total_forecast=total_forecast,
        daily_forecast=daily_forecast,
        weekly_averages=weekly_avg if request.prediction_type == "monthly" else None,
        monthly_averages=monthly_avg if request.prediction_type == "weekly" else None,
        confidence_score=round(confidence, 3),
        trend_direction=trend,
        message=f"Enhanced prediction with {confidence:.1%} confidence based on {len(request.history)} daily sales records."
    )

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
