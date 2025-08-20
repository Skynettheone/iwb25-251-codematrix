import React, { useState, useEffect } from 'react';
import { XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import '../styles/dashboard.css';

const InventoryForecast = () => {
  const [scope, setScope] = useState('single');
  const [selectedProduct, setSelectedProduct] = useState('');
  const [predictionPeriod, setPredictionPeriod] = useState(7);
  const [nonSeasonalProducts, setNonSeasonalProducts] = useState([]);
  const [loading, setLoading] = useState(false);
  const [productsLoading, setProductsLoading] = useState(true);
  const [forecastData, setForecastData] = useState(null);
  const [error, setError] = useState('');
  const [selectedProductDetail, setSelectedProductDetail] = useState(null);
  const [showProductDetail, setShowProductDetail] = useState(false);

  useEffect(() => {
    const fetchProducts = async () => {
      setProductsLoading(true);
      setError('');
      try {
        // go through ballerina which proxies to analytics
        const response = await fetch('http://localhost:9090/api/analytics/products/non_seasonal');
        if (!response.ok) throw new Error('Failed to fetch product list');
        const data = await response.json();
        setNonSeasonalProducts(data.data || []);
      } catch (err) {
        setError('Could not load products. Please ensure the Analytics Service is running.');
        console.error('Failed to fetch products:', err);
      } finally {
        setProductsLoading(false);
      }
    };
    fetchProducts();
  }, []);

  const generateForecast = async () => {
    if (scope === 'single' && !selectedProduct) {
      setError('Please select a product to forecast.');
      return;
    }
    setLoading(true);
    setError('');
    setForecastData(null);

    try {
      let url;
      if (scope === 'single') {
        // use backend proxy for prediction
        url = `http://localhost:9090/api/analytics/predict/inventory/${selectedProduct}`;
      } else {
        url = `http://localhost:9090/api/analytics/predict/inventory/all`;
      }

             const response = await fetch(url, {
         method: 'POST',
         headers: { 'Content-Type': 'application/json' },
         body: JSON.stringify({ 
           days_to_predict: predictionPeriod,
           prediction_type: "weekly",
           product_id: scope === 'single' ? selectedProduct : undefined
         })
       });

      if (!response.ok) {
        let errorMessage = 'Forecast generation failed.';
        try {
          const errData = await response.json();
          errorMessage = errData.detail || errData.message || errorMessage;
        } catch (parseError) {
          errorMessage = `HTTP ${response.status}: ${response.statusText}`;
        }
        throw new Error(errorMessage);
      }
      
      const data = await response.json();
      
      // validate response format
      if (scope === 'single') {
        if (!data.product_name || data.total_forecast === undefined) {
          console.error('Invalid single product response:', data);
          throw new Error('Invalid response format for single product forecast.');
        }
      } else {
        if (!Array.isArray(data)) {
          console.error('Invalid batch response:', data);
          throw new Error('Invalid response format for batch forecast.');
        }
        if (data.length === 0) {
          console.warn('Batch prediction returned empty array');
        }
      }
      
      setForecastData(data);

    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };
  
  const getConfidenceClass = (score) => {
    if (score > 0.8) return 'high';
    if (score > 0.6) return 'medium';
    return 'low';
  };

  const handleProductClick = (product) => {
    setSelectedProductDetail(product);
    setShowProductDetail(true);
  };

  const handleBackToAllProducts = () => {
    setShowProductDetail(false);
    setSelectedProductDetail(null);
  };

  return (
    <div className="inventory-forecast">
      <div className="page-header">
        <div>
          <h1 className="page-title">Inventory Forecast</h1>
          <p className="page-description">Generate ML-powered demand forecasts for non-seasonal items.</p>
        </div>
      </div>
      
      <div className="prediction-card">
        <div className="card-header">
            <h3>Forecast Configuration</h3>
        </div>
        <div className="forecast-controls">
            <div className="control-group">
            <label>Forecast Scope</label>
            <select value={scope} onChange={(e) => setScope(e.target.value)} className="form-select">
                <option value="single">Single Product</option>
                <option value="all">All Products</option>
            </select>
            </div>

            {scope === 'single' && (
            <div className="control-group">
            <label>Product</label>
            <select value={selectedProduct} onChange={(e) => setSelectedProduct(e.target.value)} className="form-select" disabled={productsLoading}>
                <option value="">
                    {productsLoading ? 'Loading products...' : 'Select a product...'}
                </option>
                {nonSeasonalProducts.map(product => (
                <option key={product.product_id} value={product.product_id}>
                    {product.name}
                </option>
                ))}
            </select>
            </div>
            )}

            <div className="control-group">
            <label>Prediction Period</label>
            <select value={predictionPeriod} onChange={(e) => setPredictionPeriod(Number(e.target.value))} className="form-select">
                <option value={7}>Next 7 Days</option>
                <option value={14}>Next 14 Days</option>
                <option value={30}>Next 30 Days</option>
            </select>
            </div>

            <button className="btn btn-primary" onClick={generateForecast} disabled={loading || (scope === 'single' && !selectedProduct) || (scope === 'all' && nonSeasonalProducts.length === 0)}>
            {loading ? <><div className="spinner"></div><span>Generating...</span></> : 'Generate Forecast'}
            </button>
        </div>
        {error && <div className="alert error">{error}</div>}
        {!productsLoading && nonSeasonalProducts.length === 0 && (
          <div className="alert warning">
            No non-seasonal products found. Please ensure products are available in the database.
          </div>
        )}
      </div>


             {forecastData && (
         <div className="result-card">
             {scope === 'single' ? (
               // single product forecast display
               <>
                 <div className="card-header">
                     <h3>Forecast Result for: {forecastData.product_name}</h3>
                     <span className={`confidence-badge ${getConfidenceClass(forecastData.confidence_score)}`}>
                         {(forecastData.confidence_score * 100).toFixed(0)}% Confidence
                     </span>
                 </div>

                 <div className="forecast-summary">
                     <div className="summary-stat">
                         <span className="stat-label">Total Forecast</span>
                         <span className="stat-value">{forecastData.total_forecast} units</span>
                     </div>
                     <div className="summary-stat">
                         <span className="stat-label">Estimated Revenue</span>
                         <span className="stat-value">LKR {forecastData.estimated_revenue?.toLocaleString() || 'N/A'}</span>
                     </div>
                     <div className="summary-stat">
                         <span className="stat-label">Trend Direction</span>
                         <span className="stat-value">{forecastData.trend_direction}</span>
                     </div>
                                           <div className="summary-stat">
                          <span className="stat-label">Best Algorithm</span>
                          <span className="stat-value">{forecastData.best_algorithm || forecastData.model_type || 'Unknown'}</span>
                      </div>
                      <div className="summary-stat">
                          <span className="stat-label">Data Points Used</span>
                          <span className="stat-value">{forecastData.data_points_used || 0}</span>
                      </div>
                      <div className="summary-stat">
                          <span className="stat-label">Trend Strength</span>
                          <span className="stat-value">{(forecastData.trend_strength * 100).toFixed(0)}%</span>
                      </div>
                      <div className="summary-stat">
                          <span className="stat-label">Seasonality Score</span>
                          <span className="stat-value">{(forecastData.seasonality_score * 100).toFixed(0)}%</span>
                      </div>
                 </div>

                 {forecastData.stock_recommendations && (
                     <div className="stock-recommendations">
                         <h4>Stock Management Recommendations</h4>
                         <div className="recommendations-grid">
                             <div className="recommendation-item">
                                 <span className="label">Average Daily Demand:</span>
                                 <span className="value">{forecastData.stock_recommendations.avg_daily_demand} units</span>
                             </div>
                             <div className="recommendation-item">
                                 <span className="label">Safety Stock:</span>
                                 <span className="value">{forecastData.stock_recommendations.safety_stock} units</span>
                             </div>
                             <div className="recommendation-item">
                                 <span className="label">Reorder Point:</span>
                                 <span className="value">{forecastData.stock_recommendations.reorder_point} units</span>
                             </div>
                             <div className="recommendation-item">
                                 <span className="label">Max Stock Level:</span>
                                 <span className="value">{forecastData.stock_recommendations.max_stock} units</span>
                             </div>
                             <div className="recommendation-item">
                                 <span className="label">Current Stock Needed:</span>
                                 <span className="value">{forecastData.stock_recommendations.current_stock_needed} units</span>
                             </div>
                         </div>
                     </div>
                                   )}

                  {forecastData.model_insights && (
                      <div className="ml-insights">
                          <h4>ML Model Insights</h4>
                          <div className="insights-grid">
                              <div className="insight-item">
                                  <span className="label">Model Quality:</span>
                                  <span className={`value quality-${forecastData.model_insights.model_quality?.toLowerCase()}`}>
                                      {forecastData.model_insights.model_quality}
                                  </span>
                              </div>
                              <div className="insight-item">
                                  <span className="label">Prediction Reliability:</span>
                                  <span className={`value reliability-${forecastData.model_insights.prediction_reliability?.toLowerCase()}`}>
                                      {forecastData.model_insights.prediction_reliability}
                                  </span>
                              </div>
                              <div className="insight-item">
                                  <span className="label">Data Sufficiency:</span>
                                  <span className={`value sufficiency-${forecastData.model_insights.data_sufficiency?.toLowerCase()}`}>
                                      {forecastData.model_insights.data_sufficiency}
                                  </span>
                              </div>
                          </div>
                          {forecastData.model_insights.recommendations && forecastData.model_insights.recommendations.length > 0 && (
                              <div className="recommendations-list">
                                  <h5>Recommendations:</h5>
                                  <ul>
                                      {forecastData.model_insights.recommendations.map((rec, index) => (
                                          <li key={index}>{rec}</li>
                                      ))}
                                  </ul>
                              </div>
                          )}
                      </div>
                  )}

                  {forecastData.chart_data && (
                      <div className="forecast-chart">
                          <h4>Demand Forecast Visualization</h4>
                          <img src={`data:image/png;base64,${forecastData.chart_data}`} alt="Demand Forecast Chart" style={{ maxWidth: '100%', borderRadius: 'var(--radius-md)' }} />
                      </div>
                  )}
               </>
             ) : showProductDetail && selectedProductDetail ? (
               // individual product detail view
               <>
                 <div className="card-header">
                     <div className="header-with-back">
                         <button className="back-button" onClick={handleBackToAllProducts}>
                             ← Back to All Products
                         </button>
                         <h3>Forecast Result for: {selectedProductDetail.product_name}</h3>
                         <span className={`confidence-badge ${getConfidenceClass(selectedProductDetail.confidence_score)}`}>
                             {(selectedProductDetail.confidence_score * 100).toFixed(0)}% Confidence
                         </span>
                     </div>
                 </div>

                 <div className="forecast-summary">
                     <div className="summary-stat">
                         <span className="stat-label">Total Forecast</span>
                         <span className="stat-value">{selectedProductDetail.total_forecast} units</span>
                     </div>
                     <div className="summary-stat">
                         <span className="stat-label">Estimated Revenue</span>
                         <span className="stat-value">LKR {selectedProductDetail.estimated_revenue?.toLocaleString() || 'N/A'}</span>
                     </div>
                     <div className="summary-stat">
                         <span className="stat-label">Trend Direction</span>
                         <span className="stat-value">{selectedProductDetail.trend_direction}</span>
                     </div>
                     <div className="summary-stat">
                         <span className="stat-label">Best Algorithm</span>
                         <span className="stat-value">{selectedProductDetail.best_algorithm || selectedProductDetail.model_type || 'Unknown'}</span>
                     </div>
                     <div className="summary-stat">
                         <span className="stat-label">Data Points Used</span>
                         <span className="stat-value">{selectedProductDetail.data_points_used || 0}</span>
                     </div>
                     <div className="summary-stat">
                         <span className="stat-label">Trend Strength</span>
                         <span className="stat-value">{(selectedProductDetail.trend_strength * 100).toFixed(0)}%</span>
                     </div>
                     <div className="summary-stat">
                         <span className="stat-label">Seasonality Score</span>
                         <span className="stat-value">{(selectedProductDetail.seasonality_score * 100).toFixed(0)}%</span>
                     </div>
                 </div>

                 {selectedProductDetail.stock_recommendations && (
                     <div className="stock-recommendations">
                         <h4>Stock Management Recommendations</h4>
                         <div className="recommendations-grid">
                             <div className="recommendation-item">
                                 <span className="label">Average Daily Demand:</span>
                                 <span className="value">{selectedProductDetail.stock_recommendations.avg_daily_demand} units</span>
                             </div>
                             <div className="recommendation-item">
                                 <span className="label">Safety Stock:</span>
                                 <span className="value">{selectedProductDetail.stock_recommendations.safety_stock} units</span>
                             </div>
                             <div className="recommendation-item">
                                 <span className="label">Reorder Point:</span>
                                 <span className="value">{selectedProductDetail.stock_recommendations.reorder_point} units</span>
                             </div>
                             <div className="recommendation-item">
                                 <span className="label">Max Stock Level:</span>
                                 <span className="value">{selectedProductDetail.stock_recommendations.max_stock} units</span>
                             </div>
                             <div className="recommendation-item">
                                 <span className="label">Current Stock Needed:</span>
                                 <span className="value">{selectedProductDetail.stock_recommendations.current_stock_needed} units</span>
                             </div>
                         </div>
                     </div>
                 )}

                 {selectedProductDetail.model_insights && (
                     <div className="ml-insights">
                         <h4>ML Model Insights</h4>
                         <div className="insights-grid">
                             <div className="insight-item">
                                 <span className="label">Model Quality:</span>
                                 <span className={`value quality-${selectedProductDetail.model_insights.model_quality?.toLowerCase()}`}>
                                     {selectedProductDetail.model_insights.model_quality}
                                 </span>
                             </div>
                             <div className="insight-item">
                                 <span className="label">Prediction Reliability:</span>
                                 <span className={`value reliability-${selectedProductDetail.model_insights.prediction_reliability?.toLowerCase()}`}>
                                     {selectedProductDetail.model_insights.prediction_reliability}
                                 </span>
                             </div>
                             <div className="insight-item">
                                 <span className="label">Data Sufficiency:</span>
                                 <span className={`value sufficiency-${selectedProductDetail.model_insights.data_sufficiency?.toLowerCase()}`}>
                                     {selectedProductDetail.model_insights.data_sufficiency}
                                 </span>
                             </div>
                         </div>
                         {selectedProductDetail.model_insights.recommendations && selectedProductDetail.model_insights.recommendations.length > 0 && (
                             <div className="recommendations-list">
                                 <h5>Recommendations:</h5>
                                 <ul>
                                     {selectedProductDetail.model_insights.recommendations.map((rec, index) => (
                                         <li key={index}>{rec}</li>
                                     ))}
                                 </ul>
                             </div>
                         )}
                     </div>
                 )}

                 {selectedProductDetail.chart_data && (
                     <div className="forecast-chart">
                         <h4>Demand Forecast Visualization</h4>
                         <img src={`data:image/png;base64,${selectedProductDetail.chart_data}`} alt="Demand Forecast Chart" style={{ maxWidth: '100%', borderRadius: 'var(--radius-md)' }} />
                     </div>
                 )}
               </>
             ) : (
               // all products forecast display
               <>
                 <div className="card-header">
                     <h3>Forecast Results for All Products</h3>
                     <span className="confidence-badge medium">
                         {Array.isArray(forecastData) ? forecastData.length : 0} Products Analyzed
                     </span>
                 </div>

                 <div className="forecast-summary">
                     <div className="summary-stat">
                         <span className="stat-label">Total Products</span>
                         <span className="stat-value">{Array.isArray(forecastData) ? forecastData.length : 0}</span>
                     </div>
                     <div className="summary-stat">
                         <span className="stat-label">Total Forecast</span>
                         <span className="stat-value">
                             {Array.isArray(forecastData) ? forecastData.reduce((sum, item) => sum + (item.total_forecast || 0), 0) : 0} units
                         </span>
                     </div>
                     <div className="summary-stat">
                         <span className="stat-label">Avg Confidence</span>
                         <span className="stat-value">
                             {Array.isArray(forecastData) && forecastData.length > 0 
                               ? ((forecastData.reduce((sum, item) => sum + (item.confidence_score || 0), 0) / forecastData.length) * 100).toFixed(0)
                               : 0}%
                         </span>
                     </div>
                 </div>

                 {Array.isArray(forecastData) && forecastData.length > 0 && (
                   <div className="all-products-results">
                     <h4>Individual Product Forecasts</h4>
                     <p className="click-hint">Click on any product card to view detailed forecast and charts</p>
                     <div className="products-grid">
                       {forecastData.map((product, index) => (
                         <div key={index} className="product-forecast-card clickable" onClick={() => handleProductClick(product)}>
                           <div className="product-header">
                             <h5>{product.product_name}</h5>
                             <div className="header-badges">
                               <span className={`confidence-badge ${getConfidenceClass(product.confidence_score)}`}>
                                 {(product.confidence_score * 100).toFixed(0)}%
                               </span>
                               {product.chart_data && (
                                 <span className="chart-badge" title="Chart available">
                                   📊
                                 </span>
                               )}
                             </div>
                           </div>
                           <div className="product-stats">
                             <div className="stat">
                               <span className="label">Forecast:</span>
                               <span className="value">{product.total_forecast} units</span>
                             </div>
                             <div className="stat">
                               <span className="label">Trend:</span>
                               <span className="value">{product.trend_direction}</span>
                             </div>
                             <div className="stat">
                               <span className="label">Seasonality:</span>
                               <span className="value">{(product.seasonality_score * 100).toFixed(0)}%</span>
                             </div>
                           </div>
                           <div className="click-indicator">
                             <span>{product.chart_data ? 'Click to view chart →' : 'Click to view details →'}</span>
                           </div>
                         </div>
                       ))}
                     </div>
                   </div>
                 )}
               </>
             )}
         </div>
       )}
    </div>
  );
};

export default InventoryForecast;