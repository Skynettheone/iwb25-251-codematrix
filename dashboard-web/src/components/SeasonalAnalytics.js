import React, { useState, useEffect } from 'react';
import '../styles/dashboard.css';

const SeasonalAnalytics = () => {
  const [seasonalData, setSeasonalData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchSeasonalData = async () => {
      setLoading(true);
      setError('');
      try {
        const response = await fetch('http://localhost:8000/analytics/seasonal');
        if (!response.ok) throw new Error('Failed to fetch seasonal analytics');
        const data = await response.json();
        setSeasonalData(data);
      } catch (err) {
        setError(err.message);
        console.error('Failed to fetch seasonal data:', err);
      } finally {
        setLoading(false);
      }
    };
    fetchSeasonalData();
  }, []);
  
  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-LK', { style: 'currency', currency: 'LKR', minimumFractionDigits: 0 }).format(amount);
  };

      // calculate days until each season
  const calculateDaysUntilSeason = (seasonName) => {
    const today = new Date();
    const currentYear = today.getFullYear();
    
    switch (seasonName.toLowerCase()) {
      case 'vesak':
              const vesakDate = new Date(currentYear, 4, 5);
        if (vesakDate < today) {
          vesakDate.setFullYear(currentYear + 1);
        }
        const vesakDays = Math.ceil((vesakDate - today) / (1000 * 60 * 60 * 24));
        console.log('Vesak days:', vesakDays);
        return vesakDays;
        
              case 'christmas':
          const christmasDate = new Date(currentYear, 11, 25);
        if (christmasDate < today) {
          christmasDate.setFullYear(currentYear + 1);
        }
        const christmasDays = Math.ceil((christmasDate - today) / (1000 * 60 * 60 * 24));
        console.log('Christmas days:', christmasDays);
        return christmasDays;
        
              case 'awurudu':
          const awuruduDate = new Date(currentYear, 3, 13);
        if (awuruduDate < today) {
          awuruduDate.setFullYear(currentYear + 1);
        }
        const awuruduDays = Math.ceil((awuruduDate - today) / (1000 * 60 * 60 * 24));
        console.log('Awurudu days:', awuruduDays);
        return awuruduDays;
        
      default:
        return 0;
    }
  };

  if (loading) {
    return (
        <div className="loading-card">
            <div className="spinner"></div>
            <p>Loading Seasonal Analytics...</p>
        </div>
    );
  }

  if (error || !seasonalData) {
    return <div className="alert error">Error loading data: {error || 'No data returned from the service.'}</div>;
  }

      // debug logging
  console.log('Seasonal data:', seasonalData);
  console.log('Vesak days:', calculateDaysUntilSeason('vesak'));
  console.log('Christmas days:', calculateDaysUntilSeason('christmas'));
  console.log('Awurudu days:', calculateDaysUntilSeason('awurudu'));

    return (
    <div className="seasonal-analytics">
        <div className="seasonal-grid">
            {/* Seasonal Overview - Left Card */}
            <div className="seasonal-overview-card">
                <div className="card-header">
                    <h3>Seasonal Overview</h3>
                </div>
                <div className="overview-stats">
                    <div className="stat-item">
                        <span className="stat-label">Total Seasonal Products</span>
                        <span className="stat-value">{seasonalData.total_seasonal_products}</span>
                    </div>
                    <div className="stat-item">
                        <span className="stat-label">Upcoming Seasons</span>
                        <span className="stat-value">{seasonalData.upcoming_seasons?.length || 0}</span>
                    </div>
                    <div className="stat-item">
                        <span className="stat-label">Seasonal Revenue Forecast</span>
                        <span className="stat-value">{formatCurrency(seasonalData.seasonal_revenue_forecast)}</span>
                    </div>
                    <div className="stat-item">
                        <span className="stat-label">Top Seasonal Categories</span>
                        <span className="stat-value">{seasonalData.top_seasonal_categories?.length || 0} categories</span>
                    </div>
                </div>
            </div>
            
            {/* Upcoming Seasons - Right Card */}
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
                            <span className="days-count">{calculateDaysUntilSeason('vesak')} days</span>
                        </div>
                    </div>
                    
                    {/* Christmas Season */}
                    <div className="season-item">
                        <div className="season-info">
                            <h4>Christmas</h4>
                        </div>
                        <div className="season-stats">
                            <span className="days-count">{calculateDaysUntilSeason('christmas')} days</span>
                        </div>
                    </div>
                    
                    {/* Awurudu Season */}
                    <div className="season-item">
                        <div className="season-info">
                            <h4>Awurudu</h4>
                        </div>
                        <div className="season-stats">
                            <span className="days-count">{calculateDaysUntilSeason('awurudu')} days</span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Analyze Specific Season - Full Width Bar */}
            <div className="season-analysis-card">
                <div className="card-header">
                    <h3>Analyze Specific Season</h3>
                </div>
                <div className="season-controls">
                    <div className="control-group">
                        <label htmlFor="seasonSelect">Select Season:</label>
                        <select 
                            id="seasonSelect" 
                            className="form-select"
                        >
                            <option value="">Choose a season...</option>
                            <option value="vesak">Vesak</option>
                            <option value="christmas">Christmas</option>
                            <option value="awurudu">Awurudu</option>
                        </select>
                    </div>
                    
                    <div className="control-group">
                        <label htmlFor="forecastDays">Forecast Days:</label>
                        <input 
                            type="number" 
                            id="forecastDays"
                            defaultValue="30"
                            min="7" 
                            max="90" 
                            className="form-input"
                        />
                    </div>
                    
                    <button className="btn btn-primary">
                        Analyze Season
                    </button>
                </div>
            </div>
        </div>
    </div>
  );
};

export default SeasonalAnalytics;