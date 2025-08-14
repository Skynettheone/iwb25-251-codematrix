import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [backendStatus, setBackendStatus] = useState('N/A');
  const [transactions, setTransactions] = useState([]); // State for transaction list
  const [isLoading, setIsLoading] = useState(false); // State for loading indicator

  // useEffect hook to fetch data when the component mounts
  useEffect(() => {
    fetchTransactions();
  }, []);

  const fetchTransactions = () => {
    setIsLoading(true); // Show loading indicator
    fetch('http://localhost:9090/api/transactions')
      .then(response => {
        if (!response.ok) {
          throw new Error('Failed to fetch transactions');
        }
        return response.json();
      })
      .then(data => {
        setTransactions(data); // store the array of transactions in state
        setIsLoading(false);
      })
      .catch(error => {
        console.error('Error fetching transactions:', error);
        setIsLoading(false);
      });
  };

  const handleStatusCheck = () => {
    setBackendStatus('Checking...');
    fetch('http://localhost:9090/api/status')
      .then(response => response.json())
      .then(data => {
        setBackendStatus(data.status);
      })
      .catch(error => {
        setBackendStatus('Failed to connect to the backend.');
      });
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Admin Dashboard</h1>
        <p>The central hub for analytics and management.</p>
      </header>
      <main className="App-main">
        <div className="status-panel">
          <button onClick={handleStatusCheck}>Check Backend Status</button>
          <p>Backend Status: <strong>{backendStatus}</strong></p>
        </div>

        <div className="transactions-panel">
          <div className="panel-header">
            <h2>Recent Transactions</h2>
            <button onClick={fetchTransactions} disabled={isLoading}>
              {isLoading ? 'Loading...' : 'Refresh Transactions'}
            </button>
          </div>
          <div className="table-container">
            <table className="transactions-table">
              <thead>
                <tr>
                  <th>Transaction ID</th>
                  <th>Customer ID</th>
                  <th>Total Amount (LKR)</th>
                  <th>Date & Time</th>
                </tr>
              </thead>
              <tbody>
                {transactions.length > 0 ? (
                  transactions.map((tx) => (
                    <tr key={tx.transaction_id}>
                      <td>{tx.transaction_id}</td>
                      <td>{tx.customer_id}</td>
                      <td className="amount">{parseFloat(tx.total_amount).toFixed(2)}</td>
                      <td>{new Date(tx.created_at).toLocaleString('en-LK')}</td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan="4">No transactions found.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;