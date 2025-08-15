import React, { useState, useEffect } from 'react';
import { XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import '../styles/dashboard.css';

const Dashboard = () => {
  const [dashboardData, setDashboardData] = useState({
    totalProducts: 0,
    totalTransactions: 0,
    totalRevenue: 0,
    growthRate: 0,
    seasonalProducts: 0,
    upcomingSeasons: [],
    topCategories: [],
    recentActivity: []
  });
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      setIsLoading(true);
      setError('');

      const [productsRes, transactionsRes, seasonalRes] = await Promise.all([
        fetch('http://localhost:9090/api/products'),
        fetch('http://localhost:9090/api/transactions/latest?limit=100'),
        fetch('http://localhost:8000/seasonal/analytics')
      ]);

      const products = await productsRes.json();
      const transactions = await transactionsRes.json();
      const seasonal = seasonalRes.ok ? await seasonalRes.json() : { 
        total_seasonal_products: 0, 
        upcoming_seasons: [], 
        top_seasonal_categories: [] 
      };

      const totalRevenue = transactions.reduce((sum, t) => sum + parseFloat(t.total_amount || 0), 0);
      const seasonalProducts = products.filter(p => p.product_id.includes('SEASONAL')).length;
      
      const growthRate = 12.5;

      const categoryCount = {};
      products.forEach(product => {
        const category = product.category || 'Other';
        categoryCount[category] = (categoryCount[category] || 0) + 1;
      });

      const topCategories = Object.entries(categoryCount)
        .map(([category, count]) => ({ category, count, revenue: count * 450 }))
        .sort((a, b) => b.count - a.count)
        .slice(0, 5);

      const recentActivity = transactions.slice(0, 10).map(t => ({
        id: t.transaction_id,
        type: 'Sale',
        amount: parseFloat(t.total_amount || 0),
        time: new Date(t.transaction_date).toLocaleTimeString(),
        customer: t.customer_id
      }));

      setDashboardData({
        totalProducts: products.length,
        totalTransactions: transactions.length,
        totalRevenue,
        growthRate,
        seasonalProducts,
        upcomingSeasons: seasonal.upcoming_seasons || [],
        topCategories,
        recentActivity
      });
    } catch (err) {
      console.error('Dashboard data fetch error:', err);
      setError('Failed to load dashboard data');
    } finally {
      setIsLoading(false);
    }
  };

  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-LK', {
      style: 'currency',
      currency: 'LKR',
      minimumFractionDigits: 0
    }).format(amount);
  };

  if (isLoading) {
    return (
      <div className="dashboard-loading">
        <div className="loading-spinner"></div>
        <p>Loading Stocast Dashboard...</p>
      </div>
    );
  }

  return (
    <div className="stocast-dashboard">
      {/* header section */}
      <div className="dashboard-header">
        <div className="header-content">
          <div className="brand-section">
            <h1 className="brand-title">Stocast</h1>
            <p className="brand-subtitle">Business Intelligence & Analytics</p>
          </div>
          <div className="header-actions">
            <div className="status-indicator">
              <div className="status-dot warning"></div>
              <span>System Checking</span>
            </div>
            <button className="refresh-btn" onClick={fetchDashboardData}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path d="M23 4v6h-6M1 20v-6h6M20.49 9A9 9 0 0 0 5.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 0 1 3.51 15"/>
              </svg>
              Refresh
            </button>
          </div>
        </div>
      </div>

      {error && (
        <div className="error-banner">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <circle cx="12" cy="12" r="10"/>
            <line x1="12" y1="8" x2="12" y2="12"/>
            <line x1="12" y1="16" x2="12.01" y2="16"/>
          </svg>
          {error}
        </div>
      )}

      {/* key metrics grid */}
      <div className="metrics-grid">
        <div className="metric-card primary">
          <div className="metric-header">
            <h3>Total Revenue</h3>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <line x1="12" y1="1" x2="12" y2="23"/>
              <path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>
            </svg>
          </div>
          <div className="metric-value">{formatCurrency(dashboardData.totalRevenue)}</div>
          <div className="metric-change positive">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/>
              <polyline points="17 6 23 6 23 12"/>
            </svg>
            +{dashboardData.growthRate}% this month
          </div>
        </div>

        <div className="metric-card secondary">
          <div className="metric-header">
            <h3>Products</h3>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>
            </svg>
          </div>
          <div className="metric-value">{dashboardData.totalProducts}</div>
          <div className="metric-detail">{dashboardData.seasonalProducts} seasonal items</div>
        </div>

        <div className="metric-card accent">
          <div className="metric-header">
            <h3>Transactions</h3>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <circle cx="9" cy="21" r="1"/>
              <circle cx="20" cy="21" r="1"/>
              <path d="M1 1h4l2.68 13.39a2 2 0 0 0 2 1.61h9.72a2 2 0 0 0 2-1.61L23 6H6"/>
            </svg>
          </div>
          <div className="metric-value">{dashboardData.totalTransactions}</div>
          <div className="metric-detail">Recent activity</div>
        </div>

        <div className="metric-card info">
          <div className="metric-header">
            <h3>Seasonal Forecast</h3>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path d="M12 2v6.5l4.24-4.24"/>
              <circle cx="12" cy="12" r="10"/>
            </svg>
          </div>
          <div className="metric-value">{dashboardData.upcomingSeasons.length}</div>
          <div className="metric-detail">Upcoming seasons</div>
        </div>
      </div>

      {/* main content grid */}
      <div className="content-grid">
        {/* Top Categories Chart */}
        <div className="chart-card">
          <div className="card-header">
            <h3>Top Product Categories</h3>
            <p>Performance by category</p>
          </div>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={dashboardData.topCategories}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis 
                  dataKey="category" 
                  stroke="#64748b" 
                  fontSize={12}
                  tick={{fill: '#64748b'}}
                />
                <YAxis 
                  stroke="#64748b" 
                  fontSize={12}
                  tick={{fill: '#64748b'}}
                />
                <Tooltip 
                  formatter={(value, name) => [value, name === 'count' ? 'Products' : 'Revenue (LKR)']}
                  contentStyle={{ 
                    backgroundColor: 'white', 
                    border: '1px solid #e2e8f0',
                    borderRadius: '8px',
                    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                  }}
                />
                <Bar 
                  dataKey="count" 
                  fill="#3b82f6" 
                  name="count" 
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* seasonal forecast */}
        <div className="chart-card">
          <div className="card-header">
            <h3>Seasonal Events</h3>
            <p>Upcoming seasonal opportunities</p>
          </div>
          <div className="seasonal-events">
            {dashboardData.upcomingSeasons.length > 0 ? (
              dashboardData.upcomingSeasons.slice(0, 4).map((season, index) => (
                <div key={index} className="seasonal-event-card">
                  <div className="event-indicator">
                    <div className={`status-dot ${season.days_until <= 30 ? 'urgent' : 'normal'}`}></div>
                  </div>
                  <div className="event-content">
                    <h4>{season.season_name}</h4>
                    <p className="event-timing">In {season.days_until} days</p>
                    <div className="event-metrics">
                      <span className="products-count">{season.products_count} products</span>
                      <span className="revenue-forecast">{formatCurrency(season.estimated_revenue)}</span>
                    </div>
                  </div>
                </div>
              ))
            ) : (
              <div className="no-data">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <circle cx="12" cy="12" r="10"/>
                  <path d="M12 6v6l4 2"/>
                </svg>
                <p>No upcoming seasonal events detected</p>
              </div>
            )}
          </div>
        </div>

        {/* recent activity */}
        <div className="activity-card">
          <div className="card-header">
            <h3>Recent Activity</h3>
            <p>Latest transactions and updates</p>
          </div>
          <div className="activity-list">
            {dashboardData.recentActivity.slice(0, 6).map((activity, index) => (
              <div key={index} className="activity-item">
                <div className="activity-icon">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    {activity.type === 'Sale' ? (
                      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/>
                    ) : (
                      <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>
                    )}
                  </svg>
                </div>
                <div className="activity-content">
                  <div className="activity-title">{activity.type}</div>
                  <div className="activity-subtitle">Customer: {activity.customer}</div>
                </div>
                <div className="activity-meta">
                  <div className="activity-amount">{formatCurrency(activity.amount)}</div>
                  <div className="activity-time">{activity.time}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
