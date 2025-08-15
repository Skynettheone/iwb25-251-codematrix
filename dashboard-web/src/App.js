import React, { useState, useEffect, useCallback } from 'react';
import './styles/dashboard.css';

const Icons = {
  Dashboard: () => (
    <svg viewBox="0 0 24 24" fill="currentColor" className="nav-item-icon">
      <path d="M3 13h8V3H3v10zm0 8h8v-6H3v6zm10 0h8V11h-8v10zm0-18v6h8V3h-8z"/>
    </svg>
  ),
  Analytics: () => (
    <svg viewBox="0 0 24 24" fill="currentColor" className="nav-item-icon">
      <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
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
      <path d="M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2M12,4A8,8 0 0,1 20,12A8,8 0 0,1 12,20A8,8 0 0,1 4,12A8,8 0 0,1 12,4M7,9A1,1 0 0,1 8,10A1,1 0 0,1 7,11A1,1 0 0,1 6,10A1,1 0 0,1 7,9M12,9A1,1 0 0,1 13,10A1,1 0 0,1 12,11A1,1 0 0,1 11,10A1,1 0 0,1 12,9M17,9A1,1 0 0,1 18,10A1,1 0 0,1 17,11A1,1 0 0,1 16,10A1,1 0 0,1 17,9M12,14L15.15,16.85L14.54,17.46L12,14.92L9.46,17.46L8.85,16.85L12,14Z"/>
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
  )
};

function App() {
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

  useEffect(() => {
    loadAllData();
    
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

  const loadAllData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const results = await Promise.allSettled([
        fetch('http://localhost:9090/api/products').then(r => r.ok ? r.json() : []),
        fetch('http://localhost:9090/api/transactions/latest?limit=20').then(r => r.ok ? r.json() : []),
        fetch('http://localhost:9090/api/products/seasonal').then(r => r.ok ? r.json() : null),
        fetch('http://localhost:8000/analytics/overview').then(r => r.ok ? r.json() : null),
        fetch('http://localhost:8000/seasonal/analytics').then(r => r.ok ? r.json() : null),
        fetch('http://localhost:8000/health').then(r => r.ok ? r.json() : null)
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
  };

  // calculate stats from real data
  const stats = React.useMemo(() => {
    const totalProducts = data.products.length;
    const totalTransactions = data.transactions.length;
    const totalRevenue = data.transactions.reduce((sum, t) => sum + parseFloat(t.total_amount || 0), 0);
    const seasonalProducts = data.seasonalData?.seasonal_products?.length || 0;
    
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
    const [predictionType, setPredictionType] = useState('single'); // 'single' or 'all'
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
        if (predictionType === 'all') {
          url = `http://localhost:8000/predict/inventory/all?days=${predictionDays}&period=${predictionPeriod}`;
        } else {
          url = `http://localhost:8000/predict/inventory/${selectedProduct}?days=${predictionDays}&period=${predictionPeriod}`;
        }

        const response = await fetch(url);
        
        if (response.ok) {
          const result = await response.json();
          setPrediction(result);
        } else {
          const error = await response.json();
          setPredictionError(error.detail || 'Prediction failed');
        }
      } catch (err) {
        setPredictionError('Connection error. Please ensure analytics service is running.');
      } finally {
        setIsLoading(false);
      }
    };

    return (
      <div className="predictions-page">
        <div className="page-header">
          <h1 className="page-title">Smart Predictions</h1>
          <p className="page-description">AI-powered inventory forecasting and demand prediction</p>
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
              >
                {isLoading ? (
                  <>
                    <div className="spinner"></div>
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Icons.Analytics />
                    Generate Prediction
                  </>
                )}
              </button>
            </div>
          </div>
          
          {predictionError && (
            <div className="alert error">
              <strong>Error:</strong> {predictionError}
            </div>
          )}
        </div>

        {/* prediction results */}
        {prediction && (
          <div className="prediction-results">
            {predictionType === 'single' ? (
              <div className="single-prediction">
                <div className="result-card">
                  <div className="card-header">
                    <h3>Forecast: {prediction.product_info?.name || selectedProduct}</h3>
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
                      <span className="stat-label">Daily Average</span>
                      <span className="stat-value">
                        {(prediction.total_forecast / predictionDays).toFixed(1)} units
                      </span>
                    </div>
                    <div className="summary-stat">
                      <span className="stat-label">Estimated Revenue</span>
                      <span className="stat-value">
                        {formatCurrency(prediction.total_forecast * (prediction.product_info?.price || 0))}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="all-predictions">
                <div className="result-card">
                  <div className="card-header">
                    <h3>All Products Forecast</h3>
                  </div>
                  
                  <div className="products-grid">
                    {prediction.products?.map((productPrediction, index) => (
                      <div key={index} className="product-prediction-item">
                        <div className="product-info">
                          <h4>{productPrediction.product_name}</h4>
                          <span className="product-category">{productPrediction.category}</span>
                        </div>
                        <div className="prediction-summary">
                          <div className="prediction-value">
                            {productPrediction.total_forecast} units
                          </div>
                          <div className="prediction-revenue">
                            {formatCurrency(productPrediction.estimated_revenue || 0)}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  // analytics Component
  const Analytics = () => {
    const [analyticsData, setAnalyticsData] = useState(null);
    const [isLoading, setIsLoading] = useState(false);

    const fetchAnalytics = async () => {
      setIsLoading(true);
      try {
        const response = await fetch('http://localhost:8000/analytics/overview');
        if (response.ok) {
          const result = await response.json();
          setAnalyticsData(result);
        }
      } catch (error) {
        console.error('Analytics fetch failed:', error);
      } finally {
        setIsLoading(false);
      }
    };

    useEffect(() => {
      fetchAnalytics();
    }, []);

    return (
      <div className="analytics-page">
        <div className="page-header">
          <h1 className="page-title">📊 Analytics</h1>
          <p className="page-description">Comprehensive business insights and data analysis</p>
          <button onClick={fetchAnalytics} className="refresh-btn">Refresh Analytics</button>
        </div>

        <div className="analytics-grid">
          <div className="analytics-card">
            <div className="card-header">
              <h3>Sales Performance</h3>
            </div>
            <div className="chart-placeholder">
              {isLoading ? (
                <div className="loading-state">
                  <div className="spinner"></div>
                  <p>Loading analytics...</p>
                </div>
              ) : analyticsData ? (
                <div className="analytics-data">
                  <div className="metric-grid">
                    <div className="metric-item">
                      <span className="metric-label">Total Revenue</span>
                      <span className="metric-value">{formatCurrency(analyticsData.total_revenue || 0)}</span>
                    </div>
                    <div className="metric-item">
                      <span className="metric-label">Total Orders</span>
                      <span className="metric-value">{analyticsData.total_orders?.toLocaleString() || 'N/A'}</span>
                    </div>
                    <div className="metric-item">
                      <span className="metric-label">Average Order Value</span>
                      <span className="metric-value">{formatCurrency(analyticsData.avg_order_value || 0)}</span>
                    </div>
                  </div>
                </div>
              ) : (
                <p>No analytics data available. Ensure the analytics service is running.</p>
              )}
            </div>
          </div>

          <div className="analytics-card">
            <div className="card-header">
              <h3>Product Performance</h3>
            </div>
            <div className="product-analytics">
              {Object.entries(
                data.products.reduce((acc, product) => {
                  const category = product.category || 'Other';
                  acc[category] = (acc[category] || 0) + 1;
                  return acc;
                }, {})
              ).map(([category, count]) => (
                <div key={category} className="category-stat">
                  <span className="category-name">{category}</span>
                  <div className="category-bar">
                    <div 
                      className="category-fill" 
                      style={{width: `${(count / stats.totalProducts) * 100}%`}}
                    ></div>
                    <span className="category-count">{count}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="analytics-card">
            <div className="card-header">
              <h3>System Health</h3>
            </div>
            <div className="health-status">
              {data.systemHealth ? (
                <div className="health-metrics">
                  <div className="health-item">
                    <span className="health-label">Analytics Service</span>
                    <span className="health-status-badge online">Online</span>
                  </div>
                  <div className="health-item">
                    <span className="health-label">Backend Connection</span>
                    <span className="health-status-badge online">Connected</span>
                  </div>
                  <div className="health-item">
                    <span className="health-label">Last Updated</span>
                    <span className="health-time">{new Date().toLocaleTimeString()}</span>
                  </div>
                </div>
              ) : (
                <div className="health-metrics">
                  <div className="health-item">
                    <span className="health-label">Analytics Service</span>
                    <span className="health-status-badge offline">Offline</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  };

  // seasonal analysis component
  const SeasonalAnalysis = () => {
    const [selectedSeason, setSelectedSeason] = useState('');
    const [seasonAnalysis, setSeasonAnalysis] = useState(null);
    const [isLoading, setIsLoading] = useState(false);

    const seasons = ['vesak', 'christmas', 'awurudu'];

    const handleSeasonAnalysis = async (season) => {
      setIsLoading(true);
      try {
        const response = await fetch(`http://localhost:8000/seasonal/analyze/${season}`);
        if (response.ok) {
          const result = await response.json();
          setSeasonAnalysis(result);
        }
      } catch (error) {
        console.error('Season analysis failed:', error);
      } finally {
        setIsLoading(false);
      }
    };

    return (
      <div className="seasonal-page">
        <div className="page-header">
          <h1 className="page-title">Seasonal Analysis</h1>
          <p className="page-description">Analyze seasonal trends and predict demand for Sri Lankan seasons</p>
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
            <div className="season-buttons">
              {seasons.map(season => (
                <button 
                  key={season}
                  className={`season-btn ${selectedSeason === season ? 'active' : ''}`}
                  onClick={() => {
                    setSelectedSeason(season);
                    handleSeasonAnalysis(season);
                  }}
                >
                  {season.charAt(0).toUpperCase() + season.slice(1)}
                </button>
              ))}
            </div>
          </div>

          {data.seasonalAnalytics?.upcoming_seasons && (
            <div className="upcoming-seasons-card">
              <div className="card-header">
                <h3>Upcoming Seasons</h3>
              </div>
              <div className="upcoming-seasons">
                {data.seasonalAnalytics.upcoming_seasons.map((season, index) => (
                  <div key={index} className="season-item">
                    <div className="season-info">
                      <h4>{season.season}</h4>
                      <p>{season.days_until} days until</p>
                    </div>
                    <div className="season-stats">
                      <span className="revenue">{formatCurrency(season.estimated_revenue || 0)}</span>
                      <span className="products">{season.product_count} products</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {seasonAnalysis && (
            <div className="season-analysis-card">
              <div className="card-header">
                <h3>{seasonAnalysis.season} Analysis</h3>
              </div>
              <div className="season-analysis-content">
                <div className="analysis-stat">
                  <span className="label">Seasonal Products Found:</span>
                  <span className="value">{seasonAnalysis.seasonal_products_count}</span>
                </div>
                
                {seasonAnalysis.seasonal_products && seasonAnalysis.seasonal_products.length > 0 && (
                  <div className="seasonal-products-list">
                    <h4>Seasonal Products:</h4>
                    {seasonAnalysis.seasonal_products.map((product, index) => (
                      <div key={index} className="seasonal-product-item">
                        <span className="product-name">{product.product_name}</span>
                        <span className="product-category">{product.category}</span>
                        <span className="product-price">{formatCurrency(product.price)}</span>
                      </div>
                    ))}
                  </div>
                )}
                
                <div className="analysis-message">
                  <p>{seasonAnalysis.message}</p>
                </div>
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
        <div className="page-header">
          <h1 className="page-title">Settings</h1>
          <p className="page-description">Configure system preferences and view system information</p>
          <button onClick={resetSettings} className="reset-btn">Reset to Defaults</button>
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
                  <option value="light">Light</option>
                  <option value="dark">Dark</option>
                  <option value="auto">Auto (System)</option>
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
      <div className="page-header">
        <h1 className="page-title">Transactions</h1>
        <p className="page-description">View and manage all sales transactions</p>
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
                  <td>{new Date(transaction.transaction_date).toLocaleDateString()}</td>
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
          <h1 className="page-title">Stocast Dashboard</h1>
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
                    {new Date(transaction.transaction_date).toLocaleDateString()}
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
      case 'analytics':
        return <Analytics />;
      case 'transactions':
        return <Transactions />;
      case 'predictions':
        return <SmartPredictions />;
      case 'seasonal':
        return <SeasonalAnalysis />;
      case 'settings':
        return <Settings />;
      default:
        return <Dashboard />;
    }
  };

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
            <div className="logo-icon">S</div>
            <span className="logo-text">Stocast</span>
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
                className={`nav-item ${currentPage === 'analytics' ? 'active' : ''}`}
                onClick={() => {
                  setCurrentPage('analytics');
                  setIsMobileMenuOpen(false);
                }}
              >
                <Icons.Analytics />
                <span>Analytics</span>
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
                <span>Seasonal Analysis</span>
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