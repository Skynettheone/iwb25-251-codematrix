# Stocast - AI-Powered Retail Management & Marketing System

A comprehensive retail management solution that combines modern POS operations with AI-driven inventory optimization and personalized customer marketing. Stocast demonstrates the power of integrating machine learning with real-time business operations.

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [System Architecture](#️-system-architecture)
- [Technology Stack](#technology-stack)
- [Quick Setup & Start](#-quick-setup--start)
- [Usage Examples](#-usage-examples)
- [Technical Implementation](#-technical-implementation)
- [Current Limitations](#-current-system-limitations)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)

## 🎯 Project Overview

Stocast is designed for small to medium-sized retail businesses, providing intelligent inventory management and customer engagement through AI-powered analytics. The core innovation lies in leveraging historical sales data to solve two critical business challenges:

- **Inventory Optimization**: Predictive models forecast optimal stock levels, reducing wastage and preventing stockouts
- **Personalized Marketing**: Customer behavior analysis enables targeted campaigns for increased loyalty and retention

## 🚀 Key Features

### 1. AI-Powered Inventory Management
- **Smart Predictions**: Advanced ML models forecast stock requirements
- **Seasonal Analytics**: Specialized analysis for seasonal products (Vesak, Christmas, Awurudu)
- **Multi-Model Ensemble**: Uses Linear Regression, Random Forest, SVR, and more
- **Dynamic Confidence Scoring**: Intelligent confidence assessment for predictions
- **Real-time Updates**: Live data integration from POS transactions

### 2. Personalized Marketing System
- **Customer Segmentation**: RFM analysis with K-means clustering
- **Campaign Management**: Create and launch targeted marketing campaigns
- **Multi-channel Support**: Email (SendGrid) and SMS (Twilio) integration
- **Personalization Engine**: Dynamic message customization per customer
- **Performance Tracking**: Campaign success metrics and analytics

### 3. Modern POS Operations
- **Cross-platform Desktop App**: JavaFX-based POS system
- **Real-time Transaction Processing**: Instant data synchronization
- **Customer Loyalty Integration**: Seamless customer identification
- **Payment Processing**: Multiple payment method support

### 4. Advanced Analytics Dashboard
- **Interactive Visualizations**: Charts and graphs for data insights
- **Real-time Metrics**: Live performance indicators
- **Seasonal Analysis**: Dedicated seasonal product analytics
- **Customer Insights**: Segmentation visualization and trends

## 🏗️ System Architecture

### Microservices Architecture
Stocast follows a modern, decoupled microservices architecture for scalability and maintainability:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   JavaFX POS    │    │  React Dashboard│    │  Ballerina API  │
│   (Client)      │    │   (Web UI)      │    │   (Gateway)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐    ┌─────────────────┐
                    │ Python Analytics│    │  Supabase DB    │
                    │   (ML Engine)   │    │  (PostgreSQL)   │
                    └─────────────────┘    └─────────────────┘
```

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **POS Client** | JavaFX (Java 17+) | Cross-platform desktop application for sales transactions |
| **Admin Dashboard** | React.js | Modern web interface for analytics and management |
| **Backend API** | Ballerina | Central API gateway and business logic orchestration |
| **Analytics Service** | Python 3.x + FastAPI | Machine learning models and data analysis |
| **Database** | Supabase (PostgreSQL) | Centralized data storage with BaaS features |

## 🚀 Quick Setup & Start

### Prerequisites
- Java 17 or higher
- Node.js 16 or higher
- Python 3.8 or higher
- Ballerina 2201.5.0 or higher

### Method 1: One-Click Setup (Recommended)

**Windows:**
```bash
# run the comprehensive setup script
setup.bat

# start all services in proper order
start_all.bat
```

**Linux/Mac:**
```bash
# make scripts executable
chmod +x setup.sh start_all.sh

# run the comprehensive setup script
./setup.sh

# start all services in proper order
./start_all.sh
```

### Method 2: Individual Component Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd Ballerina_O2(backup)
```

2. **Backend Setup (Ballerina)**
```bash
cd backend-ballerina
# Windows
setup_and_run.bat
# Linux/Mac
./setup_and_run.sh
```

3. **Analytics Service Setup (Python)**
```bash
cd analytics-python
python -m venv venv
# Windows
venv\Scripts\activate.bat
# Linux/Mac
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

4. **Dashboard Setup (React)**
```bash
cd dashboard-web
npm install
npm start
```

5. **POS Client Setup (JavaFX)**
```bash
cd post-client-javafx
mvn clean install
mvn javafx:run
```

### Method 3: Docker Compose (Advanced)

```bash
# build and start all services
docker-compose up --build

# run in background
docker-compose up -d --build
```

### Method 4: Manual Step-by-Step

**Startup Order (Important):**
1. **Ballerina Backend** (Port 9090)
2. **Python Analytics** (Port 8000)
3. **React Dashboard** (Port 3000)
4. **JavaFX POS Client**

**Terminal 1 - Backend:**
```bash
cd backend-ballerina
bal run
```

**Terminal 2 - Analytics:**
```bash
cd analytics-python
source venv/bin/activate  # or venv\Scripts\activate.bat on Windows
python app.py
```

**Terminal 3 - Dashboard:**
```bash
cd dashboard-web
npm start
```

**Terminal 4 - POS Client:**
```bash
cd post-client-javafx
mvn javafx:run
```

### Service URLs
- **Backend API**: http://localhost:9090
- **Analytics Service**: http://localhost:8000
- **Dashboard**: http://localhost:3000
- **POS Client**: Desktop application

### Configuration
The setup scripts automatically create the required `Config.toml` file with:
- Database connection (Supabase)
- SendGrid API key (email notifications)
- Twilio credentials (SMS notifications)

For production, update the configuration with your own credentials.

## 📝 Usage Examples

### Inventory Prediction
1. Navigate to "Smart Predictions" in the dashboard
2. Select a product or choose "All Products"
3. Set prediction parameters (days, confidence level)
4. View AI-generated forecasts with confidence scores

### Seasonal Analytics
1. Go to "Seasonal Analytics" section
2. Select a season (Vesak, Christmas, Awurudu)
3. View historical performance and predictions
4. Analyze seasonal trends and recommendations

### Marketing Campaigns
1. Access "Marketing" section
2. Create a new campaign with target segments
3. Customize message templates
4. Launch campaign and track performance

## 🔧 Technical Implementation

### Ballerina Backend (API Gateway)
Ballerina serves as the central orchestration layer, chosen for its:

**Strengths:**
- Native HTTP/HTTPS support with built-in security
- Strong typing and compile-time error detection
- Excellent database connectivity
- Built-in concurrency model
- Network-aware design for microservices

**Usage in Stocast:**
- REST API endpoints for all client applications
- Database connection management and query execution
- Business logic implementation and validation
- Integration with Python analytics service
- Authentication and authorization handling
- Third-party service integration (SendGrid, Twilio)

**Challenges Faced:**
- **Slow Compilation**: Large codebases can have extended compilation times
- **Random Compilation Errors**: Occasional type inference issues requiring explicit type annotations
- **Limited Ecosystem**: Fewer third-party libraries compared to other languages
- **Learning Curve**: Team needed time to adapt to Ballerina's unique syntax and concepts

### Python Analytics Service (ML Engine)
The analytics service implements sophisticated machine learning capabilities:

**ML Models Implemented:**
- **Ensemble Learning**: Linear Regression, Ridge, Lasso, SVR, Random Forest, Gradient Boosting, Extra Trees, Neural Networks (MLP)
- **Clustering**: K-Means (customer segmentation)
- **Anomaly Detection**: Isolation Forest
- **Feature Engineering**: Standard Scaler, Label Encoder, Polynomial Features, SelectKBest

**Feature Engineering:**
- Lag variables and moving averages
- Exponential moving averages
- Trend and volatility indicators
- Cyclical encoding for time series
- Seasonal decomposition
- Interaction features
- Z-score normalization

**Advanced Capabilities:**
- Dynamic model selection based on data characteristics
- Intelligent confidence scoring
- Multi-panel chart generation
- Seasonal pattern detection
- Anomaly detection using Isolation Forest

### React Dashboard
Modern web interface built with React.js featuring:
- Responsive design with modern UI/UX
- Real-time data visualization
- Interactive charts and graphs
- Tabbed navigation for different analytics
- Form validation and error handling

## 📊 Current System Limitations

### Data Limitations
- **Limited Historical Data**: Stocast currently uses sample data for demonstration
- **Small Dataset Size**: ML models would benefit from larger, real-world datasets
- **Data Quality**: Need for more diverse and realistic transaction patterns
- **Seasonal Data**: Limited historical seasonal data for comprehensive analysis

### Technical Limitations
- **Authentication**: Currently using mock authentication for development
- **Third-party Integrations**: SendGrid and Twilio are mocked for demo purposes
- **Scalability**: Stocast designed for PoC, needs optimization for production
- **Error Handling**: Basic error handling, needs more robust implementation

### ML Model Limitations
- **Training Data**: Models trained on synthetic data, may not reflect real-world patterns
- **Feature Engineering**: Could benefit from more domain-specific features
- **Model Validation**: Limited cross-validation due to small dataset
- **Real-time Learning**: Models don't currently update with new data automatically

## 🔮 Future Enhancements

### Short-term
- Real data integration with POS systems
- Enhanced authentication (JWT)
- Production-ready SendGrid/Twilio integrations
- Improved error handling and performance optimization

### Medium-term
- **Advanced ML**: Real-time model retraining, A/B testing, fraud detection, new product forecasting
- **Enhanced Analytics**: Customer lifetime value, churn prediction, product recommendations, price optimization
- **Scalability**: Docker containerization, load balancing, message queuing, Redis caching

### Long-term
- **AI Features**: Computer vision inventory tracking, NLP for feedback, automated pricing, predictive maintenance
- **Integrations**: Third-party POS APIs, e-commerce platforms, accounting software, supply chain management
- **Business Intelligence**: Real-time dashboards, automated reporting, competitive analysis, market trend prediction

## 🤝 Contributing

Stocast is a proof-of-concept project demonstrating AI integration in retail systems. For production use, consider:

1. **Security Review**: Implement proper security measures
2. **Performance Testing**: Load testing and optimization
3. **Data Privacy**: GDPR and data protection compliance
4. **Scalability Planning**: Architecture for growth

## 📄 License

This project is developed as a proof-of-concept for educational and demonstration purposes.

---

*Stocast demonstrates the potential of AI in retail management. The POS system is included to show the complete data flow, but the architecture is designed to integrate with any existing POS system through API endpoints.*
