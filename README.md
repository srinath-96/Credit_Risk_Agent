# Credit_Risk_Agent
# Credit Risk Analysis Application

This project is a web application featuring a chatbot interface designed to assist with credit risk analysis. It utilizes a Python backend (Flask, smolagents, H2O) for processing and modeling, and a React frontend for user interaction.

## Features

*   **Chatbot Interface:** Interact with the application through a chat interface to request analysis.
*   **H2O AutoML Modeling:** Automatically trains and evaluates machine learning models for default prediction using H2O's AutoML.
*   **Dataset Upload:** Allows users to upload their own CSV datasets for analysis.
*   **Data Visualization (Planned/In Progress):** Functionality to generate visualizations based on the dataset.
*   **Backend API:** Provides endpoints for chat interaction, model training, and data upload.

## Project Structure
```plaintext
AcreditRiskAPP/
├── .gitignore                      # Git ignore file (recommended)
├── README.md                       # This file
│
├── backend/
│   ├── api.py                      # Main Flask application file
│   ├── ModelingTool.py             # Tool for H2O AutoML modeling
│   ├── DataVisualizationTool.py    # Tool for data visualization (if created)
│   ├── requirements.txt            # Python dependencies (to be created)
│   ├── .env                        # Environment variables (API Keys, etc.)
│   ├── best_model/                 # Directory where trained models are saved
│   ├── datasets/                   # Directory where uploaded datasets are saved
│   ├── generated_plots/            # Directory for saved visualizations
│   └── .venv/                      # Python virtual environment (optional but recommended)
│
└── frontend/
    ├── public/                     # Static assets
    │   └── index.html              # Main HTML page
    ├── src/                        # Frontend source code
    │   ├── App.jsx                 # Main React application component
    │   └── main.jsx                # React application entry point
    ├── package.json                # Frontend dependencies and scripts
    ├── package-lock.json           # Lockfile for frontend dependencies
    └── vite.config.js              # Vite build configuration
```

## Prerequisites

Before you begin, ensure you have the following installed:

*   **Python:** Version 3.10 or higher recommended.
*   **Node.js & npm:** Node.js version 18 or higher recommended (comes with npm).
*   **Java:** Required by H2O. Version 8, 11, or 17 are typically compatible. Check H2O documentation for specific requirements based on the version used. [H2O Requirements](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/welcome.html#requirements)
*   **Git:** For cloning the repository.
*   **Gemini API Key:** You need an API key for the Gemini model used by the backend.

## Installation & Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/srinath-96/Credit_Risk_Agent.git
    cd Credit_Risk_Agent
    ```

2.  **Backend Setup:**
    *   Navigate to the backend directory:
        ```bash
        cd backend
        ```
    *   Create and activate a Python virtual environment:
        ```bash
        # Create environment
        python -m venv .venv

        # Activate environment (example commands)
        # Windows (Git Bash/PowerShell)
        # source .venv/Scripts/activate
        # macOS/Linux
        source .venv/bin/activate
        ```
    *   **(Important)** Create a `requirements.txt` file. You need to list all your Python dependencies. Based on our conversations, this should include at least:
        ```
        Flask
        flask_cors
        python-dotenv
        pandas
        numpy
        h2o
        smol-agents
        # Add any other specific libraries used (e.g., ucimlrepo if still used)
        # Add specific versions if necessary, e.g., Flask==2.3.0
        ```
        *You can generate this automatically after installing packages manually:* `pip freeze > requirements.txt`
    *   Install the dependencies:
        ```bash
        pip install -r requirements.txt
        ```
    *   Create a `.env` file in the `backend` directory:
        ```dotenv
        # backend/.env
        GEMINI_API_KEY=YOUR_GEMINI_API_KEY_HERE
        ```
        Replace `YOUR_GEMINI_API_KEY_HERE` with your actual key.
    *   Ensure the `best_model`, `datasets`, and `generated_plots` directories exist within the `backend` folder. If not, create them:
        ```bash
        mkdir best_model datasets generated_plots
        ```

3.  **Frontend Setup:**
    *   Navigate to the frontend directory from the project root:
        ```bash
        cd ../frontend
        # Or if already in backend: cd ../frontend
        ```
    *   Install Node.js dependencies:
        ```bash
        npm install
        ```

## Running the Application

You need to run both the backend and frontend servers concurrently.

1.  **Start the Backend Server:**
    *   Open a terminal or command prompt.
    *   Navigate to the `backend` directory.
    *   Activate the virtual environment (if not already active):
        ```bash
        # Windows (Git Bash/PowerShell)
        # source .venv/Scripts/activate
        # macOS/Linux
        source .venv/bin/activate
        ```
    *   Run the Flask application:
        ```bash
        python api.py
        # Or using Flask CLI: flask run
        ```
    *   The backend server should typically start on `http://127.0.0.1:5001` (or the port specified in `api.py`).

2.  **Start the Frontend Server:**
    *   Open a *separate* terminal or command prompt.
    *   Navigate to the `frontend` directory.
    *   Run the Vite development server:
        ```bash
        npm run dev
        ```
    *   The frontend server should typically start on `http://localhost:5173` (check the terminal output).

3.  **Access the Application:**
    *   Open your web browser and navigate to the address provided by the frontend server (usually `http://localhost:5173`).
    *   This is What it should look like:
![image](https://github.com/user-attachments/assets/5a7ef2d2-1481-4539-b1b8-5bba63751098)
![image](https://github.com/user-attachments/assets/a578fb62-9e97-408c-9c36-5d25868fb873)
![image](https://github.com/user-attachments/assets/58e7c3ab-0df7-4108-9b87-c5affe3d5e8c)

## API Endpoints (Backend)

*   `POST /api/chat`: Handles chatbot messages, routes requests to appropriate tools (modeling, visualization).
*   `POST /api/upload-dataset`: Accepts CSV file uploads and saves them to the `backend/datasets` directory.

## Environment Variables

*   `GEMINI_API_KEY`: **Required**. Your API key for accessing the Google Gemini large language model used by the `smolagents` backend. Set this in the `backend/.env` file.

## Key Dependencies

*   **Backend:** Flask, python-dotenv, pandas, h2o, smol-agents (see `backend/requirements.txt` for full list).
*   **Frontend:** React, Vite (see `frontend/package.json` for full list).

---

Remember to replace placeholder text like `<your-repository-url>` and ensure the `requirements.txt` file is accurate for your backend setup.
