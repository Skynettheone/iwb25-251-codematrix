import React, { useState, useEffect } from 'react';
import { XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import '../styles/dashboard.css';

const Dashboard = () => {
  const [dashboardData, setDashboardData] = useState({
    totalProducts: 0,
    totalTransactions: 0,
    totalRevenue: 0,
    topCategories: [],
    recentActivity: []
  });
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');

  const fetchDashboardData = async () => {
    try {
      setIsLoading(true);
      setError('');

      // call the single, efficient overview endpoint and the latest transactions endpoint
      const [overviewRes, transactionsRes] = await Promise.all([
        fetch('http://localhost:9090/api/analytics/overview'),
        fetch('http://localhost:9090/api/transactions?limit=5') // Fetch only 5 for display
      ]);

      if (!overviewRes.ok || !transactionsRes.ok) {
        throw new Error('Failed to fetch dashboard data.');
      }

      const overviewData = await overviewRes.json();
      const transactionsData = await transactionsRes.json();
      
      const recentActivity = transactionsData.map(t => ({
        id: t.transaction_id,
        type: 'Sale',
        amount: parseFloat(t.total_amount || 0),
        time: new Date(t.created_at).toLocaleTimeString('en-LK'),
        customer: t.customer_id
      }));

      setDashboardData({
        totalProducts: overviewData.totalProducts,
        totalTransactions: overviewData.totalTransactions,
        totalRevenue: overviewData.totalRevenue,
        topCategories: overviewData.topCategories.map(c => ({...c, revenue: c.count * 450})),
        recentActivity: recentActivity
      });

    } catch (err) {
      console.error('Dashboard data fetch error:', err);
      setError('Failed to load dashboard data. Please ensure backend services are running.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-LK', {
      style: 'currency',
      currency: 'LKR',
      minimumFractionDigits: 0
    }).format(amount);
  };

  if (isLoading) {
    return (
      <div className="loading-container">
        <div className="spinner large"></div>
        <p>Loading Dashboard...</p>
      </div>
    );
  }

  return (
    <div className="stocast-dashboard">
      <div className="page-header">
        <div>
          <h1 className="page-title">Dashboard Overview</h1>
          <p className="page-description">Real-time summary of your business performance.</p>
        </div>
        <div className="header-actions">
           <button className="btn btn-primary" onClick={fetchDashboardData}>Refresh Data</button>
        </div>
      </div>
      
      {error && <div className="alert error">{error}</div>}

      <div className="stats-grid">
        <div className="stat-card revenue">
            <span className="stat-label">Total Revenue</span>
            <div className="stat-value">{formatCurrency(dashboardData.totalRevenue)}</div>
        </div>
        <div className="stat-card">
            <span className="stat-label">Total Products</span>
            <div className="stat-value">{dashboardData.totalProducts}</div>
        </div>
        <div className="stat-card">
            <span className="stat-label">Total Transactions</span>
            <div className="stat-value">{dashboardData.totalTransactions}</div>
        </div>
      </div>

      <div className="content-grid">
        <div className="content-card">
          <div className="card-header">
            <h3>Top Product Categories</h3>
            <span className="card-subtitle">By number of products</span>
          </div>
          <div className="chart-container" style={{height: '300px'}}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={dashboardData.topCategories} layout="vertical" margin={{ top: 5, right: 20, left: 30, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" hide />
                <YAxis dataKey="category" type="category" width={80} tick={{ fontSize: 12 }} />
                <Tooltip formatter={(value) => [value, 'Products']} />
                <Bar dataKey="count" fill="#10b981" background={{ fill: '#f1f5f9' }} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="content-card">
            <div className="card-header">
                <h3>Recent Activity</h3>
                <span className="card-subtitle">Latest sales transactions</span>
            </div>
            <div className="transaction-list">
                {dashboardData.recentActivity.map((activity) => (
                <div key={activity.id} className="transaction-item">
                    <div className="transaction-info">
                        <span className="transaction-id">ID: {activity.id}</span>
                        <span className="transaction-date">Customer: {activity.customer}</span>
                    </div>
                    <div className="transaction-amount">{formatCurrency(activity.amount)}</div>
                </div>
                ))}
            </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;