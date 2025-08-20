import React, { useState, useEffect, useCallback } from 'react';
import './styles/dashboard.css';
import NotificationSystem from './components/NotificationSystem';
import { useAuth } from './auth/AuthContext';
import Login from './auth/Login';
import Signup from './auth/Signup';

const Icons = {
  Dashboard: () => (
    <svg viewBox="0 0 24 24" fill="currentColor" className="nav-item-icon">
      <path d="M3 13h8V3H3v10zm0 8h8v-6H3v6zm10 0h8V11h-8v10zm0-18v6h8V3h-8z"/>
    </svg>
  ),

  Transactions: () => (
    <svg viewBox="0 0 24 24" fill="currentColor" className="nav-item-icon">
      <path d="M20 4H4c-1.11 0-1.99.89-1.99 2L2 18c0 1.11.89 2 2 2h16c1.11 0 2-.89 2-2V6c0-1.11-.89-2-2-2zm0 14H4v-6h16v6zm0-10H4V6h16v2z"/>
    </svg>
  ),
  Predictions: () => (
    <svg viewBox="0 0 24 24" fill="currentColor" className="nav-item-icon">
      <path d="M12 2C13.1 2 14 2.9 14 4C14 5.1 13.1 6 12 6C10.9 6 10 5.1 10 4C10 2.9 10.9 2 12 2M21 9V7L15 1H5C3.89 1 3 1.89 3 3V19C3 20.1 3.9 21 5 21H11V19H5V3H13V9H21Z M12 18C12.55 18 13 18.45 13 19S12.55 20 12 20 11 19.55 11 19 11.45 18 12 18M18 15.5V14C18 13.45 17.55 13 17 13S16 13.45 16 14V15.5C15.42 15.5 15 15.92 15 16.5V20.5C15 21.08 15.42 21.5 16 21.5H18C18.58 21.5 19 21.08 19 20.5V16.5C19 15.92 18.58 15.5 18 15.5Z"/>
    </svg>
  ),
  Seasonal: () => (
    <svg viewBox="0 0 24 24" fill="currentColor" className="nav-item-icon">
      <path d="M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2M12,4A8,8 0 0,1 20,12A8,8 0 0,1 12,20A8,8 0 0,1 4,12A8,8 0 0,1 12,4M7,7.5A1.5,1.5 0 0,1 8.5,9A1.5,1.5 0 0,1 7,10.5A1.5,1.5 0 0,1 5.5,9A1.5,1.5 0 0,1 7,7.5M12,7.5A1.5,1.5 0 0,1 13.5,9A1.5,1.5 0 0,1 12,10.5A1.5,1.5 0 0,1 10.5,9A1.5,1.5 0 0,1 12,7.5M17,7.5A1.5,1.5 0 0,1 18.5,9A1.5,1.5 0 0,1 17,10.5A1.5,1.5 0 0,1 15.5,9A1.5,1.5 0 0,1 17,7.5M7,13.5A1.5,1.5 0 0,1 8.5,15A1.5,1.5 0 0,1 7,16.5A1.5,1.5 0 0,1 5.5,15A1.5,1.5 0 0,1 7,13.5M12,13.5A1.5,1.5 0 0,1 13.5,15A1.5,1.5 0 0,1 12,16.5A1.5,1.5 0 0,1 10.5,15A1.5,1.5 0 0,1 12,13.5M17,13.5A1.5,1.5 0 0,1 18.5,15A1.5,1.5 0 0,1 17,16.5A1.5,1.5 0 0,1 15.5,15A1.5,1.5 0 0,1 17,13.5Z"/>
    </svg>
  ),
  TrendUp: () => (
    <svg viewBox="0 0 24 24" fill="currentColor" width="16" height="16">
      <path d="M16,6L18.29,8.29L13.41,13.17L9.41,9.17L2,16.59L3.41,18L9.41,12L13.41,16L19.71,9.71L22,12V6H16Z"/>
    </svg>
  ),
  TrendDown: () => (
    <svg viewBox="0 0 24 24" fill="currentColor" width="16" height="16">
      <path d="M22,18V12H16L18.29,14.29L13.41,9.41L9.41,13.41L2,6L3.41,4.59L9.41,10.59L13.41,6.59L19.71,12.89L22,10.59V18Z"/>
    </svg>
  ),
  Settings: () => (
    <svg viewBox="0 0 24 24" fill="currentColor" className="nav-item-icon">
      <path d="M12,15.5A3.5,3.5 0 0,1 8.5,12A3.5,3.5 0 0,1 12,8.5A3.5,3.5 0 0,1 15.5,12A3.5,3.5 0 0,1 12,15.5M19.43,12.97C19.47,12.65 19.5,12.33 19.5,12C19.5,11.67 19.47,11.34 19.43,11L21.54,9.37C21.73,9.22 21.78,8.95 21.66,8.73L19.66,5.27C19.54,5.05 19.27,4.96 19.05,5.05L16.56,6.05C16.04,5.66 15.5,5.32 14.87,5.07L14.5,2.42C14.46,2.18 14.25,2 14,2H10C9.75,2 9.54,2.18 9.5,2.42L9.13,5.07C8.5,5.32 7.96,5.66 7.44,6.05L4.95,5.05C4.73,4.96 4.46,5.05 4.34,5.27L2.34,8.73C2.22,8.95 2.27,9.22 2.46,9.37L4.57,11C4.53,11.34 4.5,11.67 4.5,12C4.5,12.33 4.53,12.65 4.57,12.97L2.46,14.63C2.27,14.78 2.22,15.05 2.34,15.27L4.34,18.73C4.46,18.95 4.73,19.03 4.95,18.95L7.44,17.94C7.96,18.34 8.5,18.68 9.13,18.93L9.5,21.58C9.54,21.82 9.75,22 10,22H14C14.25,22 14.46,21.82 14.5,21.58L14.87,18.93C15.5,18.68 16.04,18.34 16.56,17.94L19.05,18.95C19.27,19.03 19.54,18.95 19.66,18.73L21.66,15.27C21.78,15.05 21.73,14.78 21.54,14.63L19.43,12.97Z"/>
    </svg>
  ),
  Menu: () => (
    <svg viewBox="0 0 24 24" fill="currentColor">
      <path d="M3,6H21V8H3V6M3,11H21V13H3V11M3,16H21V18H3V16Z"/>
    </svg>
  ),
  Close: () => (
    <svg viewBox="0 0 24 24" fill="currentColor">
      <path d="M19,6.41L17.59,5L12,10.59L6.41,5L5,6.41L10.59,12L5,17.59L6.41,19L12,13.41L17.59,19L19,17.59L13.41,12L19,6.41Z"/>
    </svg>
  ),
  Notifications: () => (
    <svg viewBox="0 0 24 24" fill="currentColor" className="nav-item-icon">
      <path d="M12,22C13.1,22 14,21.1 14,20H10C10,21.1 10.9,22 12,22M18,16V11C18,7.93 16.37,5.36 13.5,4.68V4C13.5,3.17 12.83,2.5 12,2.5S10.5,3.17 10.5,4V4.68C7.64,5.36 6,7.92 6,11V16L4,18V19H20V18L18,16M16,17H8V11C8,8.52 9.51,6.5 12,6.5S16,8.52 16,11V17Z"/>
    </svg>
  ),
  Logout: () => (
    <svg viewBox="0 0 24 24" fill="currentColor" className="nav-item-icon">
      <path d="M17,7L15.59,8.41L18.17,11H8V13H18.17L15.59,15.59L17,17L22,12M4,5H12V3H4C2.89,3 2,3.89 2,5V19A2,2 0 0,0 4,21H12V19H4V5Z"/>
    </svg>
  )
};

function App() {
  const { token, role, user, logout } = useAuth();
  const [currentPage, setCurrentPage] = useState('dashboard');
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [theme, setTheme] = useState(() => {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) return savedTheme;
    
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      return 'dark';
    }
    return 'light';
  });
  const [data, setData] = useState({
    products: [],
    transactions: [],
    seasonalData: null,
    seasonalAnalytics: null,
    systemHealth: null,
    analytics: null
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

      // auth gate: if not logged in, show login/signup
  const [authMode, setAuthMode] = useState('login');

  useEffect(() => {
    localStorage.setItem('theme', theme);
    
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handleSystemThemeChange = (e) => {
      if (theme === 'auto') {
        document.documentElement.setAttribute('data-theme', e.matches ? 'dark' : 'light');
      }
    };
    
    if (theme === 'auto') {
      document.documentElement.setAttribute('data-theme', mediaQuery.matches ? 'dark' : 'light');
      mediaQuery.addListener(handleSystemThemeChange);
    }
    
    const handleClickOutside = (event) => {
      if (isMobileMenuOpen && !event.target.closest('.sidebar') && !event.target.closest('.mobile-menu-toggle')) {
        setIsMobileMenuOpen(false);
      }
    };

    const handleEscapeKey = (event) => {
      if (event.key === 'Escape' && isMobileMenuOpen) {
        setIsMobileMenuOpen(false);
      }
    };

    if (isMobileMenuOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = 'unset';
    }

    document.addEventListener('click', handleClickOutside);
    document.addEventListener('keydown', handleEscapeKey);

    return () => {
      document.removeEventListener('click', handleClickOutside);
      document.removeEventListener('keydown', handleEscapeKey);
      if (theme === 'auto') {
        mediaQuery.removeListener(handleSystemThemeChange);
      }
      document.body.style.overflow = 'unset';
    };
  }, [isMobileMenuOpen, theme]);

  const loadAllData = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      const authHeader = token ? { Authorization: `Bearer ${token}` } : {};
      const results = await Promise.allSettled([
        fetch('http://localhost:9090/api/products').then(r => r.ok ? r.json() : []),
        fetch('http://localhost:9090/api/transactions/latest?limit=20').then(r => r.ok ? r.json() : []),
        fetch('http://localhost:9090/api/products/seasonal').then(r => r.ok ? r.json() : null),
        fetch('http://localhost:8000/analytics/overview').then(r => r.ok ? r.json() : null),
        fetch('http://localhost:8000/seasonal/analytics').then(r => r.ok ? r.json() : null),
        fetch('http://localhost:8000/health', { headers: authHeader }).then(r => r.ok ? r.json() : null)
      ]);

      setData({
        products: results[0].status === 'fulfilled' ? results[0].value : [],
        transactions: results[1].status === 'fulfilled' ? results[1].value : [],
        seasonalData: results[2].status === 'fulfilled' ? results[2].value : null,
        analytics: results[3].status === 'fulfilled' ? results[3].value : null,
        seasonalAnalytics: results[4].status === 'fulfilled' ? results[4].value : null,
        systemHealth: results[5].status === 'fulfilled' ? results[5].value : null
      });

    } catch (err) {
      setError('Failed to load data');
      console.error('Error loading data:', err);
    } finally {
      setLoading(false);
    }
  }, [token]);

      // load data on mount and when auth token changes
  useEffect(() => {
    loadAllData();
  }, [loadAllData]);

  // calculate stats from real data
  const stats = React.useMemo(() => {
    const totalProducts = data.products.length;
    const totalTransactions = data.transactions.length;
    const totalRevenue = data.transactions.reduce((sum, t) => sum + parseFloat(t.total_amount || 0), 0);
    const seasonalProducts = Array.isArray(data.seasonalData?.data)
      ? data.seasonalData.data.length
      : (Array.isArray(data.seasonalData) ? data.seasonalData.length : 0);
    
    return {
      totalProducts,
      totalTransactions, 
      totalRevenue,
      seasonalProducts
    };
  }, [data]);

  // currency support 
  const formatCurrency = (amount, currency = 'LKR') => {
    const currencyMap = {
      'LKR': { symbol: 'LKR', locale: 'en-LK' },
      'USD': { symbol: 'USD', locale: 'en-US' },
      'EUR': { symbol: 'EUR', locale: 'en-EU' }
    };

    const currencyInfo = currencyMap[currency] || currencyMap['LKR'];
    
    return new Intl.NumberFormat(currencyInfo.locale, {
      style: 'currency',
      currency: currencyInfo.symbol,
      minimumFractionDigits: 0
    }).format(amount);
  };

  const handleThemeChange = (newTheme) => {
    setTheme(newTheme);
    
    if (newTheme === 'auto') {
      const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      document.documentElement.setAttribute('data-theme', isDark ? 'dark' : 'light');
    } else {
      document.documentElement.setAttribute('data-theme', newTheme);
    }
    
    localStorage.setItem('theme', newTheme);
  };

  const SmartPredictions = () => {
    const [predictionType, setPredictionType] = useState('single');
    const [selectedProduct, setSelectedProduct] = useState('');
    const [predictionDays, setPredictionDays] = useState(7);
    const [predictionPeriod, setPredictionPeriod] = useState('weekly');
    const [prediction, setPrediction] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [predictionError, setPredictionError] = useState(null);

    const handlePrediction = async () => {
      if (predictionType === 'single' && !selectedProduct) {
        setPredictionError('Please select a product');
        return;
      }

      setIsLoading(true);
      setPredictionError(null);
      
      try {
        let url;
        let requestBody;
        
        if (predictionType === 'all') {
          // route through ballerina backend for orchestration
          url = `http://localhost:9090/api/analytics/predict/inventory/all`;
          requestBody = {
            days: predictionDays,
            period: predictionPeriod
          };
        } else {
          // get product details first
          const productDetails = data.products.find(p => p.product_id === selectedProduct);
          if (!productDetails) {
            setPredictionError('Product not found');
            return;
          }
          
          let historyData = null;
          try {
            const historyResponse = await fetch(`http://localhost:9090/api/analytics/product/sales_history?product_id=${selectedProduct}`);
            if (historyResponse.ok) {
              historyData = await historyResponse.json();
            } else {
              console.warn('history fetch failed, proceeding without history');
            }
          } catch (e) {
            console.warn('history fetch errored, proceeding without history');
          }
          url = `http://localhost:9090/api/analytics/predict/inventory/${selectedProduct}`;
          requestBody = {
            days_to_predict: parseInt(predictionDays),
            prediction_type: predictionPeriod
          };
        }

        const response = await fetch(url, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(requestBody)
        });
        
        if (response.ok) {
          const result = await response.json();
          setPrediction(result);
        } else {
          const error = await response.json();
          setPredictionError(error.detail || error.message || 'Prediction failed');
        }
      } catch (err) {
        setPredictionError('Connection error. Please ensure analytics service is running.');
      } finally {
        setIsLoading(false);
      }
    };

    return (
      <div className="predictions-page">
        <div className="dashboard-header">
          <div>
            <h1 className="page-title">Smart Predictions</h1>
            <p className="page-description">AI-powered inventory forecasting and demand prediction</p>
          </div>
          <div className="header-actions">
            <div className="status-indicator">
              <div className={`status-dot ${data.systemHealth?.status === 'healthy' ? 'healthy' : 'warning'}`}></div>
              <span>System {data.systemHealth?.status === 'healthy' ? 'Operational' : 'Checking'}</span>
            </div>
          </div>
        </div>

        <div className="prediction-card">
          <div className="card-header">
            <h3>Prediction Configuration</h3>

          </div>
          
          <div className="prediction-controls">
            {/* prediction type selection */}
            <div className="control-group full-width">
              <label>Prediction Type</label>
              <div className="radio-group">
                <label className="radio-label">
                  <input 
                    type="radio" 
                    name="predictionType" 
                    value="single"
                    checked={predictionType === 'single'}
                    onChange={(e) => setPredictionType(e.target.value)}
                  />
                  <span>Single Product</span>
                </label>
                <label className="radio-label">
                  <input 
                    type="radio" 
                    name="predictionType" 
                    value="all"
                    checked={predictionType === 'all'}
                    onChange={(e) => setPredictionType(e.target.value)}
                  />
                  <span>All Products</span>
                </label>
              </div>
            </div>

            <div className="control-row">
              {/* product selection - only show for single product */}
              {predictionType === 'single' && (
                <div className="control-group">
                  <label>Select Product</label>
                  <select 
                    value={selectedProduct} 
                    onChange={(e) => setSelectedProduct(e.target.value)}
                    className="form-select"
                  >
                    <option value="">Choose a product...</option>
                    {data.products.map(product => (
                      <option key={product.product_id} value={product.product_id}>
                        {product.name} ({product.category})
                      </option>
                    ))}
                  </select>
                </div>
              )}
              
              <div className="control-group">
                <label>Forecast Days</label>
                <input 
                  type="number" 
                  value={predictionDays}
                  onChange={(e) => setPredictionDays(parseInt(e.target.value) || 7)}
                  min="1" 
                  max="30"
                  className="form-input"
                />
              </div>
              
              <div className="control-group">
                <label>Period Type</label>
                <select 
                  value={predictionPeriod}
                  onChange={(e) => setPredictionPeriod(e.target.value)}
                  className="form-select"
                >
                  <option value="weekly">Weekly</option>
                  <option value="monthly">Monthly</option>
                </select>
              </div>
              
              <button 
                className="btn btn-primary"
                onClick={handlePrediction}
                disabled={isLoading || (predictionType === 'single' && !selectedProduct)}
                style={{opacity: 1, cursor: 'pointer'}}
              >
                {isLoading ? (
                  <>
                    <div className="spinner"></div>
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Icons.Predictions />
                    Generate Prediction
                  </>
                )}
              </button>
            </div>

            <span className="prediction-errorraised">* In first time if a error raised try second time it will work.</span>
            
            <p className="prediction-example-small">
              <em>When you select "Chicken Kottu" for 7 days with weekly period, you'll see: Total forecast of 35 units, 85% confidence score, trend direction, estimated revenue of LKR 17,500, daily breakdown, stock recommendations, and interactive chart.</em>
            </p>
          </div>
          
          {predictionError && (
            <div className="alert error">
              <strong>Error:</strong> {predictionError}
            </div>
          )}
        </div>

        {prediction && (
          <div className="prediction-results">
            {predictionType === 'single' ? (
              <div className="single-prediction">
                <div className="result-card">
                  <div className="card-header">
                    <h3>Forecast: {prediction.product_name || selectedProduct}</h3>
                    <span className={`confidence-badge ${prediction.confidence_score > 0.8 ? 'high' : prediction.confidence_score > 0.6 ? 'medium' : 'low'}`}>
                      {(prediction.confidence_score * 100).toFixed(1)}% Confidence
                    </span>
                  </div>
                  
                  <div className="forecast-chart">
                    <h4>Daily Sales Forecast</h4>
                    <div className="chart-bars">
                      {prediction.daily_forecast?.map((value, index) => {
                        const maxValue = Math.max(...prediction.daily_forecast);
                        const height = maxValue > 0 ? (value / maxValue) * 100 : 0;
                        return (
                          <div key={index} className="chart-bar">
                            <div 
                              className="bar-fill"
                              style={{height: `${height}%`}}
                            >
                              <span className="bar-value">{value}</span>
                            </div>
                            <span className="bar-label">Day {index + 1}</span>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  <div className="forecast-summary">
                    <div className="summary-stat">
                      <span className="stat-label">Total Forecast</span>
                      <span className="stat-value">{prediction.total_forecast} units</span>
                    </div>
                    <div className="summary-stat">
                      <span className="stat-label">Confidence Score</span>
                      <span className="stat-value">{(prediction.confidence_score * 100).toFixed(1)}%</span>
                    </div>
                    <div className="summary-stat">
                      <span className="stat-label">Trend Direction</span>
                      <span className="stat-value">{prediction.trend_direction || 'stable'}</span>
                    </div>
                    <div className="summary-stat">
                      <span className="stat-label">Estimated Revenue</span>
                      <span className="stat-value">
                        {formatCurrency(prediction.estimated_revenue || (prediction.total_forecast * (prediction.product_price || 0)))}
                      </span>
                    </div>
                    <div className="summary-stat">
                      <span className="stat-label">Best Algorithm</span>
                      <span className="stat-value">{prediction.best_algorithm || prediction.model_type || 'ML Model'}</span>
                    </div>
                    <div className="summary-stat">
                      <span className="stat-label">Data Points Used</span>
                      <span className="stat-value">{prediction.data_points_used || 'N/A'}</span>
                    </div>
                  </div>

                  {/* Stock Recommendations */}
                  {prediction.stock_recommendations && (
                    <div className="stock-recommendations">
                      <h4>Stock Recommendations</h4>
                      <div className="recommendations-grid">
                        <div className="recommendation-item">
                          <span className="label">Safety Stock</span>
                          <span className="value">{prediction.stock_recommendations.safety_stock || 0} units</span>
                        </div>
                        <div className="recommendation-item">
                          <span className="label">Reorder Point</span>
                          <span className="value">{prediction.stock_recommendations.reorder_point || 0} units</span>
                        </div>
                        <div className="recommendation-item">
                          <span className="label">Max Stock</span>
                          <span className="value">{prediction.stock_recommendations.max_stock || 0} units</span>
                        </div>
                        <div className="recommendation-item">
                          <span className="label">Current Stock Needed</span>
                          <span className="value">{prediction.stock_recommendations.current_stock_needed || 0} units</span>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* ML Insights */}
                  {prediction.model_insights && (
                    <div className="ml-insights">
                      <h4>ML Model Insights</h4>
                      <div className="insights-grid">
                        <div className="insight-item">
                          <span className="label">Model Quality</span>
                          <span className={`value quality-${prediction.model_insights.model_quality?.toLowerCase() || 'unknown'}`}>
                            {prediction.model_insights.model_quality || 'Unknown'}
                          </span>
                        </div>
                        <div className="insight-item">
                          <span className="label">Prediction Reliability</span>
                          <span className={`value reliability-${prediction.model_insights.prediction_reliability?.toLowerCase() || 'unknown'}`}>
                            {prediction.model_insights.prediction_reliability || 'Unknown'}
                          </span>
                        </div>
                        <div className="insight-item">
                          <span className="label">Data Sufficiency</span>
                          <span className={`value sufficiency-${prediction.model_insights.data_sufficiency?.toLowerCase() || 'unknown'}`}>
                            {prediction.model_insights.data_sufficiency || 'Unknown'}
                          </span>
                        </div>
                      </div>
                      {prediction.model_insights.recommendations && (
                        <div className="recommendations-list">
                          <h5>Recommendations:</h5>
                          <ul>
                            {prediction.model_insights.recommendations.map((rec, index) => (
                              <li key={index}>{rec}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Chart Display */}
                  {prediction.chart_data && (
                    <div className="prediction-chart">
                      <h4>Demand Forecast Visualization</h4>
                      <img 
                        src={`data:image/png;base64,${prediction.chart_data}`} 
                        alt="Demand Forecast Chart" 
                        style={{ maxWidth: '100%', borderRadius: 'var(--radius-md)' }}
                      />
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div className="all-predictions">
                <div className="result-card">
                  <div className="card-header">
                    <h3>All Products Forecast</h3>
                  </div>
                  
                  <div className="products-grid">
                    {Array.isArray(prediction) ? prediction.map((productPrediction, index) => (
                      <div key={index} className="product-prediction-item">
                        <div className="product-info">
                          <h4>{productPrediction.product_name}</h4>
                          <span className="product-category">{productPrediction.category}</span>
                        </div>
                        <div className="prediction-summary">
                          <div className="prediction-value">
                            {productPrediction.total_forecast} units
                          </div>
                          <div className="prediction-confidence">
                            {(productPrediction.confidence_score * 100).toFixed(1)}% confidence
                          </div>
                          <div className="prediction-trend">
                            Trend: {productPrediction.trend_direction || 'stable'}
                          </div>
                          <div className="prediction-revenue">
                            {formatCurrency(productPrediction.estimated_revenue || (productPrediction.total_forecast * 100))}
                          </div>
                          {productPrediction.stock_recommendations && (
                            <div className="stock-info">
                              <small>Safety: {productPrediction.stock_recommendations.safety_stock || 0} | Reorder: {productPrediction.stock_recommendations.reorder_point || 0}</small>
                            </div>
                          )}
                        </div>
                      </div>
                    )) : (
                      <div className="no-predictions">
                        <p>No prediction data available</p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

      // analytics component


  // seasonal analysis component
  const SeasonalAnalysis = () => {
    const [selectedSeason, setSelectedSeason] = useState('');
    const [seasonAnalysis, setSeasonAnalysis] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [forecastDays, setForecastDays] = useState(30);

    const seasons = ['vesak', 'christmas', 'awurudu'];

    const handleSeasonAnalysis = async (season) => {
      setIsLoading(true);
      try {
        const response = await fetch(`http://localhost:8000/seasonal/analyze/${season}?forecast_days=${forecastDays}`);
        if (response.ok) {
          const result = await response.json();
          setSeasonAnalysis(result);
        } else {
          console.error('Season analysis failed:', response.status);
        }
      } catch (error) {
        console.error('Season analysis failed:', error);
      } finally {
        setIsLoading(false);
      }
    };

    return (
      <div className="seasonal-page">
        <div className="dashboard-header">
          <div>
            <h1 className="page-title">Seasonal Analysis</h1>
            <p className="page-description">Analyze seasonal trends and predict demand for Sri Lankan seasons</p>
          </div>
          <div className="header-actions">
            <div className="status-indicator">
              <div className={`status-dot ${data.systemHealth?.status === 'healthy' ? 'healthy' : 'warning'}`}></div>
              <span>System {data.systemHealth?.status === 'healthy' ? 'Operational' : 'Checking'}</span>
            </div>
          </div>
        </div>

        <div className="seasonal-grid">
          <div className="seasonal-overview-card">
            <div className="card-header">
              <h3>Seasonal Overview</h3>
            </div>
            <div className="seasonal-overview">
              {data.seasonalAnalytics ? (
                <div className="seasonal-stats">
                  <div className="stat-item">
                    <span className="stat-label">Total Seasonal Products</span>
                    <span className="stat-value">{data.seasonalAnalytics.total_seasonal_products}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Upcoming Seasons</span>
                    <span className="stat-value">{data.seasonalAnalytics.upcoming_seasons?.length || 0}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Seasonal Revenue Forecast</span>
                    <span className="stat-value">{formatCurrency(data.seasonalAnalytics.seasonal_revenue_forecast || 0)}</span>
                  </div>
                </div>
              ) : (
                <p>Loading seasonal overview...</p>
              )}
            </div>
          </div>

          <div className="season-selector-card">
            <div className="card-header">
              <h3>Analyze Specific Season</h3>
            </div>
            <div className="prediction-controls">
              <div className="control-group">
                <label>Forecast Period (Days)</label>
                <input 
                  type="number" 
                  value={forecastDays}
                  onChange={(e) => setForecastDays(parseInt(e.target.value) || 30)}
                  min="7" 
                  max="90"
                  className="form-input"
                />
              </div>
            </div>
            <div className="season-buttons">
              {seasons.map(season => (
                <button 
                  key={season}
                  className={`season-btn ${selectedSeason === season ? 'active' : ''}`}
                  onClick={() => setSelectedSeason(season)}
                >
                  {season.charAt(0).toUpperCase() + season.slice(1)}
                </button>
              ))}
            </div>
            
            {selectedSeason && !seasonAnalysis && (
              <div className="analyze-button-container">
                <button 
                  className="btn btn-primary"
                  onClick={() => handleSeasonAnalysis(selectedSeason)}
                  disabled={isLoading}
                >
                  {isLoading ? (
                    <>
                      <div className="spinner"></div>
                      Analyzing...
                    </>
                  ) : (
                                         <>
                       <Icons.Predictions />
                       Analyze {selectedSeason.charAt(0).toUpperCase() + selectedSeason.slice(1)} Season
                     </>
                  )}
                </button>
              </div>
            )}
          </div>

          {data.seasonalAnalytics && (
            <div className="upcoming-seasons-card">
              <div className="card-header">
                <h3>Upcoming Seasons</h3>
              </div>
              <div className="upcoming-seasons">
                {/* Vesak Season */}
                <div className="season-item">
                  <div className="season-info">
                    <h4>Vesak</h4>
                  </div>
                  <div className="season-stats">
                    <span className="days-count">{(() => {
                      const today = new Date();
                      const currentYear = today.getFullYear();
                      const vesakDate = new Date(currentYear, 4, 5); // May 5th
                      if (vesakDate < today) {
                        vesakDate.setFullYear(currentYear + 1);
                      }
                      return Math.ceil((vesakDate - today) / (1000 * 60 * 60 * 24));
                    })()} days</span>
                  </div>
                </div>
                
                {/* Christmas Season */}
                <div className="season-item">
                  <div className="season-info">
                    <h4>Christmas</h4>
                  </div>
                  <div className="season-stats">
                    <span className="days-count">{(() => {
                      const today = new Date();
                      const currentYear = today.getFullYear();
                      const christmasDate = new Date(currentYear, 11, 25); // December 25th
                      if (christmasDate < today) {
                        christmasDate.setFullYear(currentYear + 1);
                      }
                      return Math.ceil((christmasDate - today) / (1000 * 60 * 60 * 24));
                    })()} days</span>
                  </div>
                </div>
                
                {/* Awurudu Season */}
                <div className="season-item">
                  <div className="season-info">
                    <h4>Awurudu</h4>
                  </div>
                  <div className="season-stats">
                    <span className="days-count">{(() => {
                      const today = new Date();
                      const currentYear = today.getFullYear();
                      const awuruduDate = new Date(currentYear, 3, 13); // April 13th
                      if (awuruduDate < today) {
                        awuruduDate.setFullYear(currentYear + 1);
                      }
                      return Math.ceil((awuruduDate - today) / (1000 * 60 * 60 * 24));
                    })()} days</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {seasonAnalysis && (
            <div className="season-analysis-card">
              <div className="card-header">
                <h3>{seasonAnalysis.season} Analysis</h3>
                <div className="header-actions">
                  <button 
                    className="btn btn-secondary"
                    onClick={() => {
                      setSeasonAnalysis(null);
                      setSelectedSeason('');
                    }}
                  >
                    ← Back to Season Selection
                  </button>
                  <span className="confidence-badge high">
                    Advanced ML-Powered Predictions
                  </span>
                </div>
              </div>
              
              {/* Summary Statistics */}
              <div className="season-summary-stats">
                <div className="summary-stat">
                  <span className="stat-label">Products Analyzed</span>
                  <span className="stat-value">{seasonAnalysis.seasonal_products_count}</span>
                </div>
                <div className="summary-stat">
                  <span className="stat-label">Total Revenue Forecast</span>
                  <span className="stat-value">{formatCurrency(seasonAnalysis.total_predicted_revenue || 0)}</span>
                </div>
                <div className="summary-stat">
                  <span className="stat-label">Average Confidence</span>
                  <span className="stat-value">{(seasonAnalysis.average_confidence * 100).toFixed(1)}%</span>
                </div>
                <div className="summary-stat">
                  <span className="stat-label">High Confidence Products</span>
                  <span className="stat-value">{seasonAnalysis.high_confidence_products}</span>
                </div>
              </div>

              {/* Seasonal Insights */}
              {seasonAnalysis.seasonal_insights && (
                <div className="seasonal-insights-section">
                  <h4>Seasonal Insights</h4>
                  <div className="insights-grid">
                    <div className="insight-item">
                      <span className="label">Peak Period</span>
                      <span className="value">{seasonAnalysis.seasonal_insights.peak_period}</span>
                    </div>
                    <div className="insight-item">
                      <span className="label">Recommended Stock</span>
                      <span className="value">{Math.round(seasonAnalysis.seasonal_insights.recommended_stock)} units</span>
                    </div>
                    <div className="insight-item">
                      <span className="label">High Demand Products</span>
                      <span className="value">{seasonAnalysis.seasonal_insights.high_demand_products}</span>
                    </div>
                    <div className="insight-item">
                      <span className="label">Revenue Potential</span>
                      <span className="value">{formatCurrency(seasonAnalysis.seasonal_insights.revenue_potential)}</span>
                    </div>
                    <div className="insight-item">
                      <span className="label">Season Strength</span>
                      <span className="value">{(seasonAnalysis.seasonal_insights.season_strength * 100).toFixed(1)}%</span>
                    </div>
                    <div className="insight-item">
                      <span className="label">Avg Daily Demand</span>
                      <span className="value">{seasonAnalysis.seasonal_insights.avg_daily_demand} units</span>
                    </div>
                  </div>
                  
                  {/* Data Quality Section */}
                  {seasonAnalysis.seasonal_insights.data_quality && (
                    <div className="data-quality-section">
                      <h5>Data Quality Analysis</h5>
                      <div className="data-quality-grid">
                        <div className="data-quality-item">
                          <span className="label">Sufficient Data Products</span>
                          <span className="value good">{seasonAnalysis.seasonal_insights.data_quality.sufficient_data_products}</span>
                        </div>
                        <div className="data-quality-item">
                          <span className="label">Insufficient Data Products</span>
                          <span className="value warning">{seasonAnalysis.seasonal_insights.data_quality.insufficient_data_products}</span>
                        </div>
                      </div>
                      
                      {/* Data Quality Warnings */}
                      {seasonAnalysis.seasonal_insights.data_quality.warnings && seasonAnalysis.seasonal_insights.data_quality.warnings.length > 0 && (
                        <div className="data-warnings">
                          <h6>⚠️ Data Quality Warnings:</h6>
                          <ul>
                            {seasonAnalysis.seasonal_insights.data_quality.warnings.map((warning, index) => (
                              <li key={index} className="warning-item">{warning}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
                
              {/* Enhanced Product Analysis */}
              {seasonAnalysis.seasonal_analysis && seasonAnalysis.seasonal_analysis.length > 0 && (
                <div className="seasonal-products-list">
                  <h4>Product Predictions</h4>
                  <div className="products-grid">
                    {seasonAnalysis.seasonal_analysis.map((analysis, index) => (
                      <div key={index} className="seasonal-product-card">
                        <div className="product-header">
                          <h5 className="product-name">{analysis.product_name}</h5>
                          <span className="product-category">{analysis.category}</span>
                        </div>
                        
                        <div className="prediction-metrics">
                          <div className="metric-row">
                            <span className="metric-label">Daily Forecast:</span>
                            <span className="metric-value">{analysis.predicted_daily} units</span>
                          </div>
                          <div className="metric-row">
                            <span className="metric-label">Total Forecast:</span>
                            <span className="metric-value">{analysis.predicted_total} units</span>
                          </div>
                          <div className="metric-row">
                            <span className="metric-label">Revenue Forecast:</span>
                            <span className="metric-value">{formatCurrency(analysis.predicted_revenue)}</span>
                          </div>
                          <div className="metric-row">
                            <span className="metric-label">Price:</span>
                            <span className="metric-value">{formatCurrency(analysis.price)}</span>
                          </div>
                        </div>
                        
                        <div className="product-analytics">
                          <div className="analytics-row">
                            <span className={`confidence-badge ${analysis.confidence > 0.7 ? 'high' : analysis.confidence > 0.4 ? 'medium' : 'low'}`}>
                              {(analysis.confidence * 100).toFixed(0)}% Confidence
                            </span>
                            <span className={`trend-badge ${analysis.trend_direction}`}>
                              {analysis.trend_direction}
                            </span>
                          </div>
                          <div className="data-info">
                            <small>Data points: {analysis.data_points_used} | Seasonality: {(analysis.seasonality_score * 100).toFixed(1)}%</small>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
                
              <div className="analysis-message">
                <p>{seasonAnalysis.message}</p>
                <small>Advanced ML analysis using ensemble learning, seasonal adjustments, and trend analysis</small>
              </div>
            </div>
          )}

          {isLoading && (
            <div className="loading-card">
              <div className="spinner"></div>
              <p>Analyzing {selectedSeason} season...</p>
            </div>
          )}
        </div>
      </div>
    );
  };

  // settings component
  const Settings = () => {
    const [settings, setSettings] = useState({
      currency: 'LKR',
      refreshInterval: 30,
      theme: theme, // Use global theme state
      dateFormat: 'DD/MM/YYYY',
      autoRefresh: true,
      analyticsEndpoint: 'http://localhost:8000',
      backendEndpoint: 'http://localhost:9090'
    });

    const [systemInfo, setSystemInfo] = useState(null);
    const [isLoading, setIsLoading] = useState(false);

    useEffect(() => {
      setSettings(prev => ({
        ...prev,
        theme: theme
      }));
    }, []); 

    const fetchSystemInfo = useCallback(async () => {
      setIsLoading(true);
      try {
        const [healthResponse, backendResponse] = await Promise.allSettled([
          fetch(`${settings.analyticsEndpoint}/health`),
          fetch(`${settings.backendEndpoint}/api/health`)
        ]);

        const healthData = healthResponse.status === 'fulfilled' && healthResponse.value.ok 
          ? await healthResponse.value.json() : null;
        const backendData = backendResponse.status === 'fulfilled' && backendResponse.value.ok 
          ? await backendResponse.value.json() : null;

        setSystemInfo({
          analytics: healthData,
          backend: backendData,
          frontend: {
            version: '1.0.0',
            build: new Date().toISOString().split('T')[0],
            status: 'running'
          }
        });
      } catch (error) {
        console.error('Failed to fetch system info:', error);
      } finally {
        setIsLoading(false);
      }
    }, [settings.analyticsEndpoint, settings.backendEndpoint]);

    useEffect(() => {
      fetchSystemInfo();
    }, [fetchSystemInfo]);

    const handleSettingChange = (key, value) => {
      setSettings(prev => ({
        ...prev,
        [key]: value
      }));
      
      if (key === 'theme') {
        handleThemeChange(value);
      }
      
      if (key === 'currency') {
        window.localStorage.setItem('preferredCurrency', value);
      }
    };

    const resetSettings = () => {
      const defaultSettings = {
        currency: 'LKR',
        refreshInterval: 30,
        theme: 'light',
        dateFormat: 'DD/MM/YYYY',
        autoRefresh: true,
        analyticsEndpoint: 'http://localhost:8000',
        backendEndpoint: 'http://localhost:9090'
      };
      setSettings(defaultSettings);
      handleThemeChange('light');
    };

    return (
      <div className="settings-page">
        <div className="dashboard-header">
          <div>
            <h1 className="page-title">Settings</h1>
            <p className="page-description">Configure system preferences and view system information</p>
          </div>
          <div className="header-actions">
            <button onClick={resetSettings} className="reset-btn">Reset to Defaults</button>
            <div className="status-indicator">
              <div className={`status-dot ${data.systemHealth?.status === 'healthy' ? 'healthy' : 'warning'}`}></div>
              <span>System {data.systemHealth?.status === 'healthy' ? 'Operational' : 'Checking'}</span>
            </div>
          </div>
        </div>

        <div className="settings-grid">
          <div className="settings-card">
            <div className="card-header">
              <h3>System Configuration</h3>
            </div>
            <div className="settings-content">
              <div className="setting-item">
                <label className="setting-label">Analytics Service Endpoint</label>
                <input 
                  type="url" 
                  value={settings.analyticsEndpoint}
                  onChange={(e) => handleSettingChange('analyticsEndpoint', e.target.value)}
                  className="setting-input"
                  placeholder="http://localhost:8000"
                />
              </div>

              <div className="setting-item">
                <label className="setting-label">Backend Service Endpoint</label>
                <input 
                  type="url" 
                  value={settings.backendEndpoint}
                  onChange={(e) => handleSettingChange('backendEndpoint', e.target.value)}
                  className="setting-input"
                  placeholder="http://localhost:9090"
                />
              </div>

              <div className="setting-item">
                <label className="setting-label">Auto Refresh Interval (seconds)</label>
                <select 
                  value={settings.refreshInterval}
                  onChange={(e) => handleSettingChange('refreshInterval', parseInt(e.target.value))}
                  className="setting-select"
                >
                  <option value={15}>15 seconds</option>
                  <option value={30}>30 seconds</option>
                  <option value={60}>1 minute</option>
                  <option value={300}>5 minutes</option>
                  <option value={600}>10 minutes</option>
                </select>
              </div>

              <div className="setting-item">
                <label className="setting-toggle">
                  <input 
                    type="checkbox" 
                    checked={settings.autoRefresh}
                    onChange={(e) => handleSettingChange('autoRefresh', e.target.checked)}
                  />
                  <span className="toggle-slider"></span>
                  Enable Auto Refresh
                </label>
              </div>
            </div>
          </div>

          <div className="settings-card">
            <div className="card-header">
              <h3>Display Preferences</h3>
            </div>
            <div className="settings-content">
              <div className="setting-item">
                <label className="setting-label">Theme</label>
                <select 
                  value={settings.theme}
                  onChange={(e) => handleSettingChange('theme', e.target.value)}
                  className="setting-select"
                >
                  <option value="light">Light Theme</option>
                  <option value="dark">Dark Theme</option>
                  <option value="auto">Auto (System Preference)</option>
                </select>
                <small className="setting-hint">Choose your preferred theme or follow system settings</small>
              </div>

              <div className="setting-item">
                <label className="setting-label">Currency Format</label>
                <select 
                  value={settings.currency}
                  onChange={(e) => handleSettingChange('currency', e.target.value)}
                  className="setting-select"
                >
                  <option value="LKR">LKR (Sri Lankan Rupee)</option>
                  <option value="USD">USD (US Dollar)</option>
                  <option value="EUR">EUR (Euro)</option>
                </select>
                <small className="setting-hint">Changes how monetary values are displayed throughout the system</small>
              </div>

              <div className="setting-item">
                <label className="setting-label">Date Format</label>
                <select 
                  value={settings.dateFormat}
                  onChange={(e) => handleSettingChange('dateFormat', e.target.value)}
                  className="setting-select"
                >
                  <option value="DD/MM/YYYY">DD/MM/YYYY (Sri Lankan Standard)</option>
                  <option value="MM/DD/YYYY">MM/DD/YYYY (US Standard)</option>
                  <option value="YYYY-MM-DD">YYYY-MM-DD (ISO Standard)</option>
                </select>
              </div>



              <div className="currency-preview">
                <h4>Currency Preview:</h4>
                <div className="preview-examples">
                  <span className="preview-item">Small Amount: {formatCurrency(1250, settings.currency)}</span>
                  <span className="preview-item">Large Amount: {formatCurrency(125000, settings.currency)}</span>
                </div>
              </div>
            </div>
          </div>

          <div className="settings-card system-info-card">
            <div className="card-header">
              <h3>System Information</h3>
              <button onClick={fetchSystemInfo} className="refresh-btn" disabled={isLoading}>
                {isLoading ? 'Loading...' : 'Refresh'}
              </button>
            </div>
            <div className="settings-content">
              {isLoading ? (
                <div className="loading-state">
                  <div className="spinner"></div>
                  <p>Loading system information...</p>
                </div>
              ) : systemInfo ? (
                <div className="system-info">
                  {/* Frontend Info */}
                  <div className="info-section">
                    <h4>Frontend (React Dashboard)</h4>
                    <div className="info-grid">
                      <div className="info-item">
                        <span className="info-label">Version</span>
                        <span className="info-value">{systemInfo.frontend.version}</span>
                      </div>
                      <div className="info-item">
                        <span className="info-label">Build Date</span>
                        <span className="info-value">{systemInfo.frontend.build}</span>
                      </div>
                      <div className="info-item">
                        <span className="info-label">Status</span>
                        <span className="status-badge online">Running</span>
                      </div>
                    </div>
                  </div>

                  <div className="info-section">
                    <h4>Analytics Service (FastAPI)</h4>
                    <div className="info-grid">
                      <div className="info-item">
                        <span className="info-label">Endpoint</span>
                        <span className="info-value">{settings.analyticsEndpoint}</span>
                      </div>
                      <div className="info-item">
                        <span className="info-label">Status</span>
                        <span className={`status-badge ${systemInfo.analytics ? 'online' : 'offline'}`}>
                          {systemInfo.analytics ? 'Online' : 'Offline'}
                        </span>
                      </div>
                      {systemInfo.analytics && (
                        <>
                          <div className="info-item">
                            <span className="info-label">Service</span>
                            <span className="info-value">{systemInfo.analytics.service || 'Smart Retail Analytics'}</span>
                          </div>
                          <div className="info-item">
                            <span className="info-label">ML Engine</span>
                            <span className="info-value">scikit-learn</span>
                          </div>
                        </>
                      )}
                    </div>
                  </div>

                  <div className="info-section">
                    <h4>Backend Service (Ballerina)</h4>
                    <div className="info-grid">
                      <div className="info-item">
                        <span className="info-label">Endpoint</span>
                        <span className="info-value">{settings.backendEndpoint}</span>
                      </div>
                      <div className="info-item">
                        <span className="info-label">Status</span>
                        <span className={`status-badge ${systemInfo.backend ? 'online' : 'offline'}`}>
                          {systemInfo.backend ? 'Online' : 'Offline'}
                        </span>
                      </div>
                      <div className="info-item">
                        <span className="info-label">Database</span>
                        <span className="info-value">PostgreSQL</span>
                      </div>
                    </div>
                  </div>

                  <div className="info-section">
                    <h4>System Statistics</h4>
                    <div className="info-grid">
                      <div className="info-item">
                        <span className="info-label">Total Products</span>
                        <span className="info-value">{stats.totalProducts}</span>
                      </div>
                      <div className="info-item">
                        <span className="info-label">Total Transactions</span>
                        <span className="info-value">{stats.totalTransactions}</span>
                      </div>
                      <div className="info-item">
                        <span className="info-label">Seasonal Products</span>
                        <span className="info-value">{stats.seasonalProducts}</span>
                      </div>
                      <div className="info-item">
                        <span className="info-label">Last Updated</span>
                        <span className="info-value">{new Date().toLocaleString()}</span>
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <p>Unable to load system information. Please check service connections.</p>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  };

  // transactions component
  const Transactions = () => (
    <div className="transactions-page">
      <div className="dashboard-header">
        <div>
          <h1 className="page-title">Transactions</h1>
          <p className="page-description">View and manage all sales transactions</p>
        </div>
        <div className="header-actions">
          <div className="status-indicator">
            <div className={`status-dot ${data.systemHealth?.status === 'healthy' ? 'healthy' : 'warning'}`}></div>
            <span>System {data.systemHealth?.status === 'healthy' ? 'Operational' : 'Checking'}</span>
          </div>
        </div>
      </div>

      <div className="transactions-table-card">
        <div className="card-header">
          <h3>Recent Transactions</h3>
          <div className="header-stats">
            <span>Total: {stats.totalTransactions}</span>
            <span>Revenue: {formatCurrency(stats.totalRevenue)}</span>
          </div>
        </div>
        
        <div className="table-container">
          <table className="transactions-table">
            <thead>
              <tr>
                <th>Transaction ID</th>
                <th>Date</th>
                <th>Customer</th>
                <th>Amount</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
        {data.transactions.map((transaction, index) => (
                <tr key={index}>
                  <td>#{transaction.transaction_id}</td>
          <td>{new Date(transaction.created_at || transaction.transaction_date || Date.now()).toLocaleDateString()}</td>
                  <td>{transaction.customer_id || 'Guest'}</td>
                  <td>{formatCurrency(transaction.total_amount)}</td>
                  <td>
                    <span className="status-badge completed">Completed</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );

  const Dashboard = () => (
    <div className="dashboard-content">
      <div className="dashboard-header">
        <div>
          <h1 className="page-title">Dashboard</h1>
          <p className="page-description">Monitor sales, inventory, and business analytics in real-time.</p>
        </div>
        <div className="header-actions">
          <div className="status-indicator">
            <div className={`status-dot ${data.systemHealth?.status === 'healthy' ? 'healthy' : 'warning'}`}></div>
            <span>System {data.systemHealth?.status === 'healthy' ? 'Operational' : 'Checking'}</span>
          </div>
        </div>
      </div>

      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-header">
            <span className="stat-label">Total Products</span>
            <div className="stat-badge">{stats.totalProducts}</div>
          </div>
          <div className="stat-value">{stats.totalProducts}</div>
          <div className="stat-change positive">
            <Icons.TrendUp />
            <span>Increased from last month</span>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-header">
            <span className="stat-label">Total Transactions</span>
            <div className="stat-badge">{stats.totalTransactions}</div>
          </div>
          <div className="stat-value">{stats.totalTransactions}</div>
          <div className="stat-change positive">
            <Icons.TrendUp />
            <span>Increased from last month</span>
          </div>
        </div>

        <div className="stat-card revenue">
          <div className="stat-header">
            <span className="stat-label">Total Revenue</span>
            <div className="stat-icon">$</div>
          </div>
          <div className="stat-value">{formatCurrency(stats.totalRevenue)}</div>
          <div className="stat-change positive">
            <Icons.TrendUp />
            <span>From all sales</span>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-header">
            <span className="stat-label">Seasonal Items</span>
            <div className="stat-badge seasonal">{stats.seasonalProducts}</div>
          </div>
          <div className="stat-value">{stats.seasonalProducts}</div>
          <div className="stat-change">
            <span>Available now</span>
          </div>
        </div>
      </div>

      <div className="content-grid">
        <div className="content-card">
          <div className="card-header">
            <h3>Recent Transactions</h3>
            <span className="card-subtitle">Latest activity</span>
          </div>
          <div className="transaction-list">
      {data.transactions.slice(0, 5).map((transaction, index) => (
              <div key={index} className="transaction-item">
                <div className="transaction-info">
                  <span className="transaction-id">#{transaction.transaction_id}</span>
                  <span className="transaction-date">
        {new Date(transaction.created_at || transaction.transaction_date || Date.now()).toLocaleDateString()}
                  </span>
                </div>
                <div className="transaction-amount">
                  {formatCurrency(transaction.total_amount)}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="content-card">
          <div className="card-header">
            <h3>Product Categories</h3>
            <span className="card-subtitle">Inventory distribution</span>
          </div>
          <div className="category-list">
            {Object.entries(
              data.products.reduce((acc, product) => {
                const category = product.category || 'Other';
                acc[category] = (acc[category] || 0) + 1;
                return acc;
              }, {})
            ).slice(0, 6).map(([category, count]) => (
              <div key={category} className="category-item">
                <span className="category-name">{category}</span>
                <div className="category-count">
                  <span>{count} items</span>
                  <div className="category-bar">
                    <div 
                      className="category-fill" 
                      style={{width: `${(count / stats.totalProducts) * 100}%`}}
                    ></div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );

  const renderPage = () => {
    if (loading) {
      return (
        <div className="loading-container">
          <div className="spinner large"></div>
          <p>Loading dashboard...</p>
        </div>
      );
    }

    switch (currentPage) {
      case 'dashboard':
        return <Dashboard />;

      case 'transactions':
        return <Transactions />;
      case 'predictions':
        return <SmartPredictions />;
      case 'seasonal':
        return <SeasonalAnalysis />;
      case 'notifications':
        return <NotificationSystem />;
      case 'settings':
        return <Settings />;
      default:
        return <Dashboard />;
    }
  };

      // if not authenticated show auth screens
  if (!token) {
    return (
      <div className="app auth">
        {authMode === 'login' 
          ? <Login onSuccess={()=>setCurrentPage('dashboard')} onSwitchToSignup={()=>setAuthMode('signup')} /> 
          : <Signup onSuccess={()=>setAuthMode('login')} onSwitchToLogin={()=>setAuthMode('login')} />}
      </div>
    );
  }

  return (
    <div className="app">
      <button 
        type="button"
        className="mobile-menu-toggle"
        onClick={(e) => {
          e.preventDefault();
          e.stopPropagation();
          setIsMobileMenuOpen(!isMobileMenuOpen);
        }}
        aria-label="Toggle mobile menu"
      >
        {isMobileMenuOpen ? <Icons.Close /> : <Icons.Menu />}
      </button>

      <div 
        className={`mobile-overlay ${isMobileMenuOpen ? 'active' : ''}`}
        onClick={() => setIsMobileMenuOpen(false)}
      ></div>

      <nav className={`sidebar ${isMobileMenuOpen ? 'mobile-open' : ''}`}>
        <div className="sidebar-header">
          <div className="logo">
            <img 
              src="/stocast_logo_long.png" 
              alt="Stocast" 
              className="sidebar-logo light-mode"
            />
            <img 
              src="/stocast_logo_long_darmode.png" 
              alt="Stocast" 
              className="sidebar-logo dark-mode"
            />
          </div>
          
        </div>

        <div className="nav-menu">
          <div className="nav-section">
            <span className="nav-section-title">MENU</span>
            <div className="nav-items">
              <button 
                type="button"
                className={`nav-item ${currentPage === 'dashboard' ? 'active' : ''}`}
                onClick={() => {
                  setCurrentPage('dashboard');
                  setIsMobileMenuOpen(false);
                }}
              >
                <Icons.Dashboard />
                <span>Dashboard</span>
              </button>
              

              
              <button 
                type="button"
                className={`nav-item ${currentPage === 'transactions' ? 'active' : ''}`}
                onClick={() => {
                  setCurrentPage('transactions');
                  setIsMobileMenuOpen(false);
                }}
              >
                <Icons.Transactions />
                <span>Transactions</span>
              </button>
              
              <button 
                type="button"
                className={`nav-item ${currentPage === 'predictions' ? 'active' : ''}`}
                onClick={() => {
                  setCurrentPage('predictions');
                  setIsMobileMenuOpen(false);
                }}
              >
                <Icons.Predictions />
                <span>Smart Predictions</span>
              </button>
              
              <button 
                type="button"
                className={`nav-item ${currentPage === 'seasonal' ? 'active' : ''}`}
                onClick={() => {
                  setCurrentPage('seasonal');
                  setIsMobileMenuOpen(false);
                }}
              >
                <Icons.Seasonal />
                <span>Seasonal Analytics</span>
              </button>
              
              <button 
                type="button"
                className={`nav-item ${currentPage === 'notifications' ? 'active' : ''}`}
                onClick={() => {
                  setCurrentPage('notifications');
                  setIsMobileMenuOpen(false);
                }}
              >
                <Icons.Notifications />
                <span>Marketing</span>
              </button>
            </div>
          </div>

          <div className="nav-section">
            <span className="nav-section-title">GENERAL</span>
            <div className="nav-items">
              <button 
                type="button"
                className={`nav-item ${currentPage === 'settings' ? 'active' : ''}`}
                onClick={() => {
                  setCurrentPage('settings');
                  setIsMobileMenuOpen(false);
                }}
              >
                <Icons.Settings />
                <span>Settings</span>
              </button>
              <button 
                type="button"
                className="nav-item"
                onClick={logout}
              >
                <Icons.Logout />
                <span>Logout</span>
              </button>
            </div>
          </div>
        </div>
      </nav>

      <main className={`main-content ${isMobileMenuOpen ? 'menu-open' : ''}`}>
        {error && (
          <div className="error-banner">
            <strong>Error:</strong> {error}
            <button onClick={loadAllData} className="retry-btn">Retry</button>
          </div>
        )}
        {renderPage()}
      </main>
    </div>
  );
}

export default App;