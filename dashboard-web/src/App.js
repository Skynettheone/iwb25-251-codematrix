import React, { useState } from 'react';
import './App.css';

function App() {
  const [backendStatus, setBackendStatus] = useState('Click the button to check status.');

  const handleStatusCheck = () => {
    setBackendStatus('Checking...');

    fetch('http://localhost:9090/api/status')
      .then(response => {
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        return response.json();
      })
      .then(data => {
        setBackendStatus(data.status);
      })
      .catch(error => {
        console.error('There was a problem with the fetch operation:', error);
        setBackendStatus('Failed to connect to the backend.');
      });
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Admin Dashboard</h1>
        <p>
          The central hub for analytics and management.
        </p>
        <button onClick={handleStatusCheck} style={{marginTop: '20px', padding: '10px 20px', fontSize: '16px'}}>
          Check Backend Status
        </button>
        <p style={{marginTop: '30px', fontSize: '18px', color: '#61dafb'}}>
          Backend Status: <strong>{backendStatus}</strong>
        </p>
      </header>
    </div>
  );
}

export default App;