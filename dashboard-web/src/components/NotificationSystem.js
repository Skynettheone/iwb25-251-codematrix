import React, { useState, useEffect } from 'react';
import './NotificationSystem.css';

const NotificationSystem = () => {
  const [campaigns, setCampaigns] = useState([]);
  const [customers, setCustomers] = useState([]);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  const [showCreateCampaign, setShowCreateCampaign] = useState(false);
  const [selectedCampaignDetails, setSelectedCampaignDetails] = useState(null);
  
  const [campaignForm, setCampaignForm] = useState({
    name: '',
    target_segment: '',
    message_template: '',
    campaign_type: 'email',
    email_subject: '',
    photo_url: ''
  });

  useEffect(() => {
    fetch('http://localhost:9090/api/health')
      .then(res => res.json())
      .then(data => {
        console.log('Service health check:', data);
        return fetch('http://localhost:9090/api/test/customers');
      })
      .then(res => res.json())
      .then(data => {
        console.log('Test customers data:', data);
        if (data.count > 0) {
          console.log('Found customers in database:', data.customers);
        } else {
          console.log('No customers found in database');
        }
        loadData();
      })
      .catch(err => {
        console.error('Service not running or test failed:', err);
        setError('Backend service is not running or database connection failed. Please start the Ballerina service.');
        setLoading(false);
      });
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      setError(null);
      

      
      const [campaignsRes, customersRes] = await Promise.all([
        fetch('http://localhost:9090/api/campaigns'),
        fetch('http://localhost:9090/api/customers')
      ]);

      console.log('Campaigns response status:', campaignsRes.status);
      console.log('Customers response status:', customersRes.status);

      if (!campaignsRes.ok || !customersRes.ok) {
        const campaignsText = await campaignsRes.text();
        const customersText = await customersRes.text();
        console.log('Campaigns error response:', campaignsText);
        console.log('Customers error response:', customersText);
        throw new Error(`Failed to fetch data. Campaigns: ${campaignsRes.status}, Customers: ${customersRes.status}`);
      }
      
      const campaignsData = await campaignsRes.json();
      const customersData = await customersRes.json();

      console.log('Campaigns data:', campaignsData);
      console.log('Customers data:', customersData);

      setCampaigns(campaignsData || []);
      setCustomers(customersData || []);

    } catch (err) {
      console.error('Load data error:', err);
      setError('Failed to load data: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateCampaign = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch('http://localhost:9090/api/campaigns', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(campaignForm)
      });

      if (response.ok) {
        setShowCreateCampaign(false);
        setCampaignForm({ name: '', target_segment: '', message_template: '', campaign_type: 'email', email_subject: '', photo_url: '' });
        loadData();
      } else {
        const errText = await response.text();
        setError(`Failed to create campaign: ${errText}`);
      }
    } catch (err) {
      setError('Error creating campaign: ' + err.message);
    }
  };

  const handleLaunchCampaign = async (campaignId, { dryRun = false } = {}) => {
    const isConfirmed = window.confirm(dryRun
      ? 'Run a dry-run? This will NOT send real notifications. It will just preview counts.'
      : 'Are you sure you want to launch this campaign? This will send notifications to all customers in the target segment.'
    );
    if (!isConfirmed) return;

    try {
  const url = `http://localhost:9090/api/campaigns/${campaignId}/launch${dryRun ? '?dry_run=true' : ''}`;
  const response = await fetch(url, { method: 'POST' });

      if (response.ok) {
        const result = await response.json();
        alert(`Campaign launched successfully! ${result.message}`);
        loadData();
      } else {
        const errText = await response.text();
        setError(`Failed to launch campaign: ${errText}`);
      }
    } catch (err) {
      setError('Error launching campaign: ' + err.message);
    }
  };

  const getSegmentCount = (segment) => {
    if (!customers || customers.length === 0) return 0;
    return customers.filter(c => c.segment === segment).length;
  };
  
  const customerSegments = [...new Set(customers.map(c => c.segment))];

  if (loading) {
    return <div className="notification-loading">Loading notification system...</div>;
  }

  return (
    <div className="notification-system">
      <div className="dashboard-header">
        <div>
          <h1 className="page-title">Marketing & Campaign Management</h1>
          <p className="page-description">Create targeted marketing campaigns and manage customer notifications</p>
        </div>
        <div className="header-actions">
          <div className="status-indicator">
            <div className="status-dot healthy"></div>
            <span>System Operational</span>
          </div>
        </div>
      </div>
      
      {error && <div className="notification-error" onClick={() => setError(null)}>Error: {error} (click to dismiss)</div>}

      <div className="dashboard-content">
        <div className="notification-actions" style={{ marginBottom: '1rem' }}>
        <button 
          className="btn btn-primary"
          onClick={() => setShowCreateCampaign(true)}
        >
          Create New Campaign
        </button>
                 <button
           className="btn btn-secondary"
           onClick={() => {
             const confirmed = window.confirm("are you sure you want to run the customer segmentation process? this will re-analyze all transactions and may take a moment.");
             if (confirmed) {
               fetch('http://localhost:9090/api/customers/segment', { method: 'POST' })
                 .then(res => res.json())
                 .then(data => {
                   alert(data.message);
                   loadData();
                 })
                 .catch(err => setError("Failed to run segmentation: " + err.message));
             }
           }}
         >
          Run AI Segmentation
        </button>
                 <button
           className="btn btn-info"
           onClick={() => {
             const confirmed = window.confirm("add sample customers for testing?");
             if (confirmed) {
               fetch('http://localhost:9090/api/customers/sample', { method: 'POST' })
                 .then(res => res.json())
                 .then(data => {
                   alert(data.message);
                   loadData();
                 })
                 .catch(err => setError("Failed to add sample data: " + err.message));
             }
           }}
         >
          Add Sample Data
        </button>
        <button
          className="btn btn-warning"
          onClick={() => {
            console.log('Manual refresh clicked');
            loadData();
          }}
        >
          Refresh Data
        </button>
      </div>

      <div className="notification-section">
        <h3>Customer Segments Overview</h3>
        <div className="segments-overview">
          {customerSegments.length > 0 ? (
            customerSegments.map(segment => (
              <div key={segment} className="segment-card" data-segment={segment}>
                <h4>{segment}</h4>
                <div className="segment-count">{getSegmentCount(segment)}</div>
                <div className="segment-percentage">
                  {customers.length > 0 ? ((getSegmentCount(segment) / customers.length) * 100).toFixed(1) : 0}%
                </div>
              </div>
            ))
          ) : (
            <div>
              <p>No customer segments found. Total customers loaded: {customers.length}</p>
              {customers.length > 0 && (
                <div>
                  <p>Customer segments: {customers.map(c => c.segment).join(', ')}</p>
                  <p>Sample customers: {customers.slice(0, 3).map(c => `${c.name} (${c.segment})`).join(', ')}</p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      <div className="notification-section">
        <h3>Marketing Campaigns</h3>
        <div className="campaigns-grid">
          {campaigns.map(campaign => (
            <div key={campaign.id} className="campaign-card">
              <div className="campaign-header">
                <h4>{campaign.name}</h4>
              </div>
                             <div className="campaign-details">
                 <div className="detail-item">
                   <strong>Target:</strong> {campaign.target_segment} ({getSegmentCount(campaign.target_segment)} customers)
                 </div>
                 <div className="detail-item">
                   <strong>Type:</strong> {campaign.campaign_type === 'email' ? 'Email Campaign' : 'SMS Campaign'}
                 </div>
                 {campaign.campaign_type === 'email' && campaign.email_subject && (
                   <div className="detail-item">
                     <strong>Subject:</strong> {campaign.email_subject}
                   </div>
                 )}
                 <div className="detail-item">
                   <strong>Created:</strong> {new Date(campaign.created_at).toLocaleDateString('en-LK')}
                 </div>
               </div>
              <div className="campaign-actions">
                  <button 
                    className="btn btn-secondary btn-sm"
                    onClick={() => handleLaunchCampaign(campaign.id, { dryRun: true })}
                  >
                    Dry-run
                  </button>
                  <button 
                    className="btn btn-success btn-sm"
                    onClick={() => handleLaunchCampaign(campaign.id)}
                  >
                    Launch Campaign
                  </button>
                <button 
                  className="btn btn-info btn-sm"
                  onClick={() => setSelectedCampaignDetails(campaign)}
                >
                  View Details
                </button>
              </div>
            </div>
          ))}
           {campaigns.length === 0 && <p>No campaigns found. Create one to get started!</p>}
        </div>
      </div>

      {showCreateCampaign && (
        <div className="modal-overlay">
          <div className="modal-content">
            <div className="modal-header">
              <h3>Create New Campaign</h3>
              <button className="modal-close" onClick={() => setShowCreateCampaign(false)}>×</button>
            </div>
            <form onSubmit={handleCreateCampaign}>
              <div className="form-group">
                <label>Campaign Name</label>
                <input type="text" value={campaignForm.name} onChange={(e) => setCampaignForm({...campaignForm, name: e.target.value})} required />
              </div>
                             <div className="form-group">
                 <label>Target Segment</label>
                 <select value={campaignForm.target_segment} onChange={(e) => setCampaignForm({...campaignForm, target_segment: e.target.value})} required >
                   <option value="">Select Segment...</option>
                   {customerSegments.map(segment => (
                     <option key={segment} value={segment}>{segment} ({getSegmentCount(segment)} customers)</option>
                   ))}
                 </select>
               </div>
               <div className="form-group">
                 <label>Campaign Type</label>
                 <select value={campaignForm.campaign_type} onChange={(e) => setCampaignForm({...campaignForm, campaign_type: e.target.value})} required >
                   <option value="email">Email Campaign</option>
                   <option value="sms">SMS Campaign</option>
                 </select>
               </div>
               {campaignForm.campaign_type === 'email' && (
                 <>
                   <div className="form-group">
                     <label>Email Subject</label>
                     <input 
                       type="text" 
                       value={campaignForm.email_subject} 
                       onChange={(e) => setCampaignForm({...campaignForm, email_subject: e.target.value})} 
                       placeholder="Enter email subject line"
                     />
                   </div>
                   <div className="form-group">
                     <label>Photo URL (Optional)</label>
                     <input 
                       type="url" 
                       value={campaignForm.photo_url} 
                       onChange={(e) => setCampaignForm({...campaignForm, photo_url: e.target.value})} 
                       placeholder="https://example.com/image.jpg"
                     />
                     <small>Add a photo URL to include in the email</small>
                   </div>
                 </>
               )}
              <div className="form-group">
                <label>Message Template</label>
                <textarea value={campaignForm.message_template} onChange={(e) => setCampaignForm({...campaignForm, message_template: e.target.value})} rows="4" placeholder="Use {name} to personalize the message for the customer." required />
              </div>
              <div className="modal-actions">
                <button type="button" onClick={() => setShowCreateCampaign(false)}>Cancel</button>
                <button type="submit" className="btn btn-primary">Create Campaign</button>
              </div>
            </form>
          </div>
        </div>
      )}

      {selectedCampaignDetails && (
        <div className="modal-overlay">
          <div className="modal-content">
            <div className="modal-header">
              <h3>Campaign Details</h3>
              <button className="modal-close" onClick={() => setSelectedCampaignDetails(null)}>×</button>
            </div>
            <div className="campaign-details-modal">
              <h4>{selectedCampaignDetails.name}</h4>
              <div className="detail-section">
                <h5>Information</h5>
                                 <div className="detail-grid">
                   <div><strong>Target Segment:</strong> {selectedCampaignDetails.target_segment}</div>
                   <div><strong>Campaign Type:</strong> {selectedCampaignDetails.campaign_type === 'email' ? 'Email Campaign' : 'SMS Campaign'}</div>
                   <div><strong>Created:</strong> {new Date(selectedCampaignDetails.created_at).toLocaleString('en-LK')}</div>
                   {selectedCampaignDetails.campaign_type === 'email' && selectedCampaignDetails.email_subject && (
                     <div><strong>Email Subject:</strong> {selectedCampaignDetails.email_subject}</div>
                   )}
                   {selectedCampaignDetails.campaign_type === 'email' && selectedCampaignDetails.photo_url && (
                     <div><strong>Photo URL:</strong> <a href={selectedCampaignDetails.photo_url} target="_blank" rel="noopener noreferrer">{selectedCampaignDetails.photo_url}</a></div>
                   )}
                 </div>
              </div>
              <div className="detail-section">
                <h5>Message Template</h5>
                <div className="message-preview">{selectedCampaignDetails.message_template}</div>
              </div>
            </div>
                     </div>
         </div>
       )}
       </div>
     </div>
   );
 };

export default NotificationSystem;