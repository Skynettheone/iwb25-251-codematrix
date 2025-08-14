import React, { useState, useEffect } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, 
  BarChart, Bar, AreaChart, Area 
} from 'recharts';
import './App.css';

function App() {
  // --- State Management ---
  const [products, setProducts] = useState([]);
  const [predictionScope, setPredictionScope] = useState('single'); // 'single' or 'all'
  const [selectedProductId, setSelectedProductId] = useState('');
  const [predictionPeriod, setPredictionPeriod] = useState('weekly'); // 'weekly' or 'monthly'
  const [predictionDays, setPredictionDays] = useState(7); // 7 or 30
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState([]); // Can hold single or multiple results
  const [activeProductData, setActiveProductData] = useState(null); // For graphs
  const [latestTransactions, setLatestTransactions] = useState([]);
  const [selectedFilter, setSelectedFilter] = useState('all'); // For filtering all items view
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  // --- Data Fetching ---
  useEffect(() => {
    fetchProducts();
    fetchLatestTransactions();
  }, []);

  useEffect(() => {
    // Update prediction days based on period selection
    if (predictionPeriod === 'weekly') {
      setPredictionDays(7);
    } else {
      setPredictionDays(30);
    }
  }, [predictionPeriod]);

  const fetchProducts = async () => {
    try {
      setError('');
      const response = await fetch('http://localhost:9090/api/products');
      if (!response.ok) {
        throw new Error(`Failed to fetch products: ${response.status}`);
      }
      const data = await response.json();
      setProducts(data);
      if (data.length > 0) {
        setSelectedProductId(data[0].product_id);
      }
      console.log('Products loaded:', data.length);
    } catch (err) {
      console.error("Failed to fetch products:", err);
      setError(`Failed to load products: ${err.message}`);
    }
  };

  const fetchLatestTransactions = async () => {
    try {
      setError('');
      const response = await fetch('http://localhost:9090/api/transactions/latest?limit=10');
      if (!response.ok) {
        throw new Error(`Failed to fetch transactions: ${response.status}`);
      }
      const data = await response.json();
      setLatestTransactions(data);
      console.log('Latest transactions loaded:', data.length);
    } catch (err) {
      console.error("Failed to fetch latest transactions:", err);
      setError(`Failed to load transactions: ${err.message}`);
    }
  };

  const handleFetchPrediction = async () => {
    if (predictionScope === 'single' && !selectedProductId) {
      setError('Please select a product for prediction');
      return;
    }

    setIsLoading(true);
    setResults([]);
    setActiveProductData(null);
    setError('');
    setSuccess('');

    try {
      if (predictionScope === 'single') {
        // Single product prediction
        const response = await fetch(
          `http://localhost:9090/api/products/${selectedProductId}/prediction?days=${predictionDays}&period=${predictionPeriod}`
        );
        
        if (!response.ok) {
          throw new Error(`Prediction failed: ${response.status}`);
        }
        
        const predictionResult = await response.json();
        setResults([predictionResult]);
        setSuccess(`Successfully generated prediction for ${predictionDays} days`);
        
        // Automatically load graphs for single product
        handleViewGraphs(predictionResult.product_id);
      } else {
        // All products prediction
        const response = await fetch('http://localhost:9090/api/products/predict/all', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            days: predictionDays,
            period: predictionPeriod
          })
        });
        
        if (!response.ok) {
          throw new Error(`Batch prediction failed: ${response.status}`);
        }
        
        const predictionResults = await response.json();
        setResults(predictionResults);
        setSelectedFilter('all');
        setSuccess(`Successfully generated predictions for ${predictionResults.length} products`);
      }
    } catch (error) {
      console.error("Failed to fetch predictions:", error);
      setError(`Prediction failed: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleViewGraphs = async (productId) => {
    try {
      setError('');
      // Fetch prediction data for the selected product
      const predictionRes = await fetch(
        `http://localhost:9090/api/products/${productId}/prediction?days=${predictionDays}&period=${predictionPeriod}`
      );
      
      if (!predictionRes.ok) {
        throw new Error(`Failed to fetch graph data: ${predictionRes.status}`);
      }
      
      const predictionData = await predictionRes.json();
      
      setActiveProductData({
        prediction: predictionData
      });
      
      console.log('Graph data loaded for product:', productId);
    } catch (error) {
      console.error("Failed to fetch graph data:", error);
      setError(`Failed to load analytics: ${error.message}`);
    }
  };

  // --- Helper Functions ---
  const getFilteredResults = () => {
    if (selectedFilter === 'all' || predictionScope === 'single') {
      return results;
    }
    return results.filter(result => result.product_id === selectedFilter);
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return '#10b981'; // Green
    if (confidence >= 0.6) return '#f59e0b'; // Orange
    return '#ef4444'; // Red
  };

  const getTrendIcon = (trend) => {
    switch (trend) {
      case 'increasing': return '📈';
      case 'decreasing': return '📉';
      default: return '➡️';
    }
  };

  const getTrendClass = (trend) => {
    switch (trend) {
      case 'increasing': return 'increasing';
      case 'decreasing': return 'decreasing';
      default: return '';
    }
  };

  // Prepare chart data with better error handling
  const preparePredictionChartData = () => {
    if (!activeProductData?.prediction?.daily_forecast) {
      console.log('No daily forecast data available');
      return [];
    }
    
    return activeProductData.prediction.daily_forecast.map((value, index) => ({
      day: `Day ${index + 1}`,
      predicted: Math.max(0, Math.round(value)), // Ensure non-negative integers
      period: predictionPeriod === 'weekly' ? 'Week' : 'Month'
    }));
  };

  const prepareAverageSalesData = () => {
    if (!activeProductData?.prediction) {
      console.log('No prediction data available for averages');
      return [];
    }
    
    const prediction = activeProductData.prediction;
    
    if (predictionPeriod === 'weekly' && prediction.monthly_averages) {
      const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
      return prediction.monthly_averages.map((avg, index) => ({
        period: months[index] || `Month ${index + 1}`,
        average: parseFloat(avg.toFixed(1))
      }));
    } else if (predictionPeriod === 'monthly' && prediction.weekly_averages) {
      return prediction.weekly_averages.map((avg, index) => ({
        period: `Week ${index + 1}`,
        average: parseFloat(avg.toFixed(1))
      }));
    }
    
    console.log('No matching average data for period:', predictionPeriod);
    return [];
  };

  const getProductName = (productId) => {
    const product = products.find(p => p.product_id === productId);
    return product ? product.name : productId;
  };

  // --- UI Rendering ---
  return (
    <div className="App">
      <header className="App-header">
        <h1>AI Inventory Forecast Dashboard</h1>
        <p>Smart Predictions for Inventory Management</p>
      </header>
      
      <main className="App-main">
        {/* Error/Success Messages */}
        {error && (
          <div className="error-message">
            ⚠️ {error}
          </div>
        )}
        
        {success && (
          <div className="success-message">
            ✅ {success}
          </div>
        )}

        {/* --- Enhanced Controls Panel --- */}
        <div className="panel controls-panel">
          <h3>⚙️ Prediction Parameters</h3>
          <div className="control-row">
            <div className="control-group">
              <label>Prediction Scope</label>
              <select 
                value={predictionScope} 
                onChange={e => {
                  setPredictionScope(e.target.value);
                  setResults([]);
                  setActiveProductData(null);
                }}
                className="enhanced-select"
              >
                <option value="single">Single Product</option>
                <option value="all">All Products</option>
              </select>
            </div>
            
            {predictionScope === 'single' && (
              <div className="control-group">
                <label>Select Product</label>
                <select 
                  value={selectedProductId} 
                  onChange={e => setSelectedProductId(e.target.value)}
                  className="enhanced-select"
                >
                  {products.length === 0 ? (
                    <option value="">Loading products...</option>
                  ) : (
                    products.map(p => (
                      <option key={p.product_id} value={p.product_id}>
                        {p.name} (ID: {p.product_id})
                      </option>
                    ))
                  )}
                </select>
              </div>
            )}
            
            <div className="control-group">
              <label>Prediction Period</label>
              <select 
                value={predictionPeriod} 
                onChange={e => {
                  setPredictionPeriod(e.target.value);
                  setResults([]);
                  setActiveProductData(null);
                }}
                className="enhanced-select"
              >
                <option value="weekly">Next Week (7 Days)</option>
                <option value="monthly">Next Month (30 Days)</option>
              </select>
            </div>
            
            <div className="control-group">
              <button 
                onClick={handleFetchPrediction} 
                disabled={isLoading || (predictionScope === 'single' && !selectedProductId)}
                className="prediction-button"
              >
                {isLoading ? (
                  <>
                    <span className="loading-spinner"></span>
                    Generating...
                  </>
                ) : (
                  'Generate Forecast'
                )}
              </button>
            </div>
          </div>
        </div>

        {/* --- Results Panel --- */}
        {results.length > 0 && (
          <div className="panel results-panel">
            <h3>📊 Forecast Results</h3>
            
            {predictionScope === 'all' && (
              <div className="filter-section">
                <label>Filter Products:</label>
                <select 
                  value={selectedFilter} 
                  onChange={e => setSelectedFilter(e.target.value)}
                  className="filter-select"
                >
                  <option value="all">Show All Products</option>
                  {results.map(result => (
                    <option key={result.product_id} value={result.product_id}>
                      {getProductName(result.product_id)}
                    </option>
                  ))}
                </select>
              </div>
            )}
            
            <div className="results-grid">
              {getFilteredResults().map(result => (
                <div key={result.product_id} className="result-card">
                  <div className="card-header">
                    <h4>{getProductName(result.product_id)}</h4>
                    <span className={`trend-indicator ${getTrendClass(result.trend_direction)}`}>
                      {getTrendIcon(result.trend_direction)} {result.trend_direction}
                    </span>
                  </div>
                  
                  <div className="card-content">
                    <div className="forecast-value">
                      <span className="number">{result.total_forecast}</span>
                      <span className="unit">units</span>
                    </div>
                    
                    <div className="prediction-details">
                      <div className="detail-item">
                        <span className="label">Period:</span>
                        <span className="value">{result.forecast_period_days} days ({result.prediction_type})</span>
                      </div>
                      
                      <div className="detail-item">
                        <span className="label">Confidence:</span>
                        <span 
                          className="value confidence-score"
                          style={{ color: getConfidenceColor(result.confidence_score) }}
                        >
                          {(result.confidence_score * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                    
                    <button 
                      onClick={() => handleViewGraphs(result.product_id)}
                      className="view-graphs-btn"
                    >
                      📊 View Analytics
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* --- Enhanced Graphs Panel --- */}
        {activeProductData && (
          <div className="panel graphs-panel">
            <h3>📈 Analytics for {getProductName(activeProductData.prediction.product_id)}</h3>
            
            <div className="analytics-summary">
              <div className="summary-card">
                <div className="summary-icon">🎯</div>
                <div className="summary-content">
                  <div className="summary-value">{activeProductData.prediction.total_forecast}</div>
                  <div className="summary-label">Predicted Sales</div>
                </div>
              </div>
              
              <div className="summary-card">
                <div className="summary-icon">📈</div>
                <div className="summary-content">
                  <div className="summary-value">{activeProductData.prediction.trend_direction}</div>
                  <div className="summary-label">Trend Direction</div>
                </div>
              </div>
              
              <div className="summary-card">
                <div className="summary-icon">🔒</div>
                <div className="summary-content">
                  <div className="summary-value">{(activeProductData.prediction.confidence_score * 100).toFixed(1)}%</div>
                  <div className="summary-label">Confidence</div>
                </div>
              </div>
            </div>

            <div className="charts-container">
              {/* Average Sales Chart */}
              {prepareAverageSalesData().length > 0 && (
                <div className="chart-wrapper">
                  <h4>
                    📊 Average Sales Across {predictionPeriod === 'weekly' ? 'Year (Monthly)' : 'Month (Weekly)'}
                  </h4>
                  <ResponsiveContainer width="100%" height={350}>
                    <BarChart data={prepareAverageSalesData()}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis 
                        dataKey="period" 
                        stroke="#64748b"
                        fontSize={12}
                      />
                      <YAxis 
                        stroke="#64748b"
                        fontSize={12}
                      />
                      <Tooltip 
                        formatter={(value) => [value, 'Average Sales']}
                        labelStyle={{ color: '#1e293b' }}
                        contentStyle={{ 
                          backgroundColor: 'white', 
                          border: '1px solid #e2e8f0',
                          borderRadius: '8px'
                        }}
                      />
                      <Legend />
                      <Bar 
                        dataKey="average" 
                        fill="#3b82f6" 
                        name="Average Sales"
                        radius={[4, 4, 0, 0]}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}

              {/* Predicted Sales Chart */}
              {preparePredictionChartData().length > 0 && (
                <div className="chart-wrapper">
                  <h4>🔮 Predicted Sales ({predictionDays} Days)</h4>
                  <ResponsiveContainer width="100%" height={350}>
                    <AreaChart data={preparePredictionChartData()}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis 
                        dataKey="day" 
                        stroke="#64748b"
                        fontSize={12}
                      />
                      <YAxis 
                        stroke="#64748b"
                        fontSize={12}
                      />
                      <Tooltip 
                        formatter={(value) => [value, 'Predicted Units']}
                        labelStyle={{ color: '#1e293b' }}
                        contentStyle={{ 
                          backgroundColor: 'white', 
                          border: '1px solid #e2e8f0',
                          borderRadius: '8px'
                        }}
                      />
                      <Legend />
                      <Area 
                        type="monotone" 
                        dataKey="predicted" 
                        stroke="#10b981" 
                        fill="#10b981"
                        fillOpacity={0.6}
                        name="Predicted Sales"
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>
          </div>
        )}

        {/* --- Latest Transactions Panel --- */}
        <div className="panel transactions-panel">
          <h3>💳 Latest Transactions</h3>
          {latestTransactions.length === 0 ? (
            <div className="error-message">
              No transactions found. Please check your database connection.
            </div>
          ) : (
            <div className="transactions-table-container">
              <table className="transactions-table">
                <thead>
                  <tr>
                    <th>Transaction ID</th>
                    <th>Customer ID</th>
                    <th>Amount</th>
                    <th>Date</th>
                  </tr>
                </thead>
                <tbody>
                  {latestTransactions.map(transaction => (
                    <tr key={transaction.transaction_id}>
                      <td className="transaction-id">{transaction.transaction_id}</td>
                      <td>{transaction.customer_id}</td>
                      <td className="amount">${parseFloat(transaction.total_amount).toFixed(2)}</td>
                      <td className="date">{new Date(transaction.transaction_date).toLocaleDateString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;