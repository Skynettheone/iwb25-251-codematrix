# AI-Powered Retail Management System

This repository contains the source code for the AI-Powered Retail Management System, a Proof of Concept designed to showcase intelligent inventory management and personalized customer marketing for small to medium-sized retail businesses.

---

## **Project Overview**

The system integrates a Point of Sale (POS) client with a powerful backend that uses AI to solve two critical business challenges:

1.  **Inventory Optimization:** Predicts optimal stock levels to minimize waste and prevent stockouts.
2.  **Personalized Marketing:** Analyzes customer behavior to enable efficient, targeted notification campaigns.

### **Architecture**

The project follows a modern, decoupled microservices architecture:

* **POS Client (JavaFX):** A cross-platform desktop application for processing sales.
* **Admin Dashboard (React):** A web-based interface for analytics and management.
* **Backend API (Ballerina):** The central API hub that handles all business logic and integrations.
* **Analytics Service (Python/Flask):** A dedicated service for AI-driven stock forecasting and customer segmentation.
* **Database (PostgreSQL on Supabase):** The central data store for all transactions and customer information.

---


## **Getting Started & Setup Instructions**

This guide explains how to set up and run the entire project.  
The process has been automated to be as simple as possible for all users (team members and evaluators).

### **Prerequisites**

Ensure you have the following software installed on your system:

- [Git](https://git-scm.com/downloads)
- [Java JDK](https://www.oracle.com/java/technologies/javase/jdk17-archive-downloads.html) (Version 17 or later)
- [Ballerina](https://ballerina.io/downloads/) (Latest version)
- [Python](https://www.python.org/downloads/) (Version 3.9 or later)
- [Node.js and npm](https://nodejs.org/)

---

## **How to Run the System**

To run the full system, start all four services in **separate terminal windows**.

---

### **1. Clone the Repository**

1.  **Backend Service (Ballerina)**
    * Navigate to the `backend-ballerina` directory.
    * On Windows: Double-click and run the `setup_and_run.bat`. On macOS/Linux: `sh setup_and_run.sh`
    * Ensure your `Config.toml` file is present after running the above.
    * Run the command: `bal run`
    * The service will be available at `http://localhost:9090`.

3. **Analytics Service (Python)**
    * Navigate to the `analytics-python` directory.
    * Create and activate a Python virtual environment:
        * macOS/Linux: `source venv/bin/activate`
        * Windows: `.\venv\Scripts\activate`
    * Install dependencies: `pip install -r requirements.txt`
    * Run the service: `flask run`
    * The service will be available at: `http://localhost:5000`

3.  **POS Client (JavaFX)**
    * Navigate to the `pos-client-javafx` directory.
    * Run the command: `mvn javafx:run`
    * The desktop application window will appear.

5. **Admin Dashboard (React)**
    * Navigate to the `dashboard-web` directory.
    * Install dependencies: `npm install`
    * Run the application: `npm start`
    * A new browser tab will open automatically at: `http://localhost:3000`

---

## **Changelog / Update Log**

* **2025-08-13:**
    * **[DONE]** Created a unified, automated setup process for all users using setup scripts.
    * **[DONE]** Updated the project README.md with the new setup instructions.

* **2025-08-12:**
    * **[DONE]** Initialized project structure and starter code for all four services.
    * **[DONE]** Established basic API communication between clients (React, JavaFX) and the Ballerina backend.
    * **[DONE]** Implemented `POST /api/transactions` endpoint in Ballerina to receive sales data.
    * **[DONE]** Updated JavaFX client to send sample transaction data to the backend.
    * **[DONE]** Set up PostgreSQL database on Supabase with initial schema and Sri Lankan demo data.
    * **[DONE]** Integrated Ballerina service with the Supabase database to persist incoming transactions.
* **[NEXT]** Implement API endpoints in Ballerina to read transaction data from the database.
