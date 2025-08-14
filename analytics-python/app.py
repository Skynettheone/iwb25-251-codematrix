from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from decimal import Decimal
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import uvicorn

# pydantic models for data validation

class TransactionItemInfo(BaseModel):
    quantity: int
    created_at: str

class PredictionRequest(BaseModel):
    product_id: str
    history: List[TransactionItemInfo]

class StockPrediction(BaseModel):
    product_id: str
    forecast_next_7_days: int
    message: str

app = FastAPI(title="Analytics Service", description="Stock prediction and sales analytics")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:9090"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API endpoints

@app.get("/status")
def get_status():
    """Health check endpoint"""
    return {"status": "Analytics service running", "version": "1.0.0"}

@app.post("/analyze/summary")
def analyze_summary(transactions: List[dict]):
    """Analyze transaction summary"""
    total_sales = sum(Decimal(tx['total_amount']) for tx in transactions)
    count = len(transactions)
    return {
        "total_sales_lkr": total_sales,
        "transaction_count": count,
        "message": f"Analyzed {count} transactions totaling LKR {total_sales}"
    }

@app.post("/predict/stock", response_model=StockPrediction)
async def predict_stock(request: Request):
    """
    Predict stock requirements for the next 7 days based on historical sales data
    """
    try:
        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="Empty request body received")
        
        data = await request.json()
        product_id = data.get("product_id")
        history_data = data.get("history", [])
        
        if not product_id:
            raise HTTPException(status_code=400, detail="product_id is required")
        
        if len(history_data) < 2:
            raise HTTPException(
                status_code=400, 
                detail="Insufficient historical data. At least 2 data points required for prediction."
            )

        df_data = []
        for item in history_data:
            df_data.append({
                'quantity': int(item.get('quantity', 0)),
                'created_at': item.get('created_at', '')
            })
        
        df = pd.DataFrame(df_data)
        df['created_at'] = pd.to_datetime(df['created_at'])
        df = df.sort_values(by='created_at')
        
        # calculate day numbers for regression
        df['day_number'] = (df['created_at'] - df['created_at'].min()).dt.days
        
        # train linear regression model
        X = df[['day_number']]
        y = df['quantity']
        model = LinearRegression()
        model.fit(X, y)

        # predict next 7 days
        last_day = df['day_number'].max()
        future_days = np.array(range(last_day + 1, last_day + 8)).reshape(-1, 1)
        predicted_quantities = model.predict(future_days)
        total_forecast = max(0, int(sum(predicted_quantities)))

        return StockPrediction(
            product_id=product_id,
            forecast_next_7_days=total_forecast,
            message=f"Prediction based on {len(history_data)} historical data points"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
