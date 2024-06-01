# ThoraxInsight

## Introduction

This project is structured into three main parts, each fulfilling a critical role in the overall functionality of the application. Here’s a quick overview of each component:

1. **Backend (Folder: `back`)**
    - The backend of this application is developed using FastAPI, a modern, fast (high-performance), web framework for building APIs with Python 3.6+ based on standard Python type hints. This component is responsible for handling all server-side logic, including API endpoints, database interactions, and server-side processing.

2. **Frontend (Folder: `front/thoraxinsight`)**
    - The frontend is built using React, a popular JavaScript library for building user interfaces. This part of the application handles the client-side rendering, providing a dynamic and interactive user experience. All the UI components, state management, and client-side routing are implemented here.

3. **Neural Network Training (Folder: `training`)**
    - This folder contains all the Python scripts necessary for training the neural network. It includes the data preprocessing scripts, model definition, training loop, and evaluation metrics. Additionally, the final trained model is exported and stored as a file, which is utilized by the backend for making predictions.

Below, you will find detailed instructions on how to set up and run each part of the project.

## Step-by-Step Guide to Run the Project

### Prerequisite: Clone the Repository

1. **Clone the repository to your local machine:**
   ```bash
   git clone https://github.com/AML4206-MINE20242/Proyecto_AML.git
   cd Proyecto_AML
   ```

### Backend (Folder: `back`)

1. **Navigate to the `back` directory:**
   ```bash
   cd back
   ```

2. **Create a virtual environment and activate it:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the FastAPI server:**
   ```bash
   uvicorn main:app --reload
   ```
   The backend server will be running at `http://127.0.0.1:8000`.

### Frontend (Folder: `front/thoraxinsight`)

1. **Navigate to the `front/thoraxinsight` directory:**
   ```bash
   cd front/thoraxinsight
   ```

2. **Install the required dependencies:**
   ```bash
   npm install
   ```

3. **Start the React development server:**
   ```bash
   npm start
   ```
   The frontend will be running at `http://localhost:3000`.


### Docker Deployment (Optional)

For a containerized deployment, you can use the provided `Dockerfile` for backend and frontend.
#### Backend and Worker

1. **Navigate to the root directory of the project (if not already there):**
   ```bash
   cd Proyecto_AML
   ```

2. **Build the Docker image for the backend and worker:**
   ```bash
   docker build -t proyecto_aml_backend_worker .
   ```

3. **Run the Docker container for the backend and worker:**
   ```bash
   docker run -p 8000:8000 proyecto_aml_backend_worker
   ```

#### Frontend

1. **Navigate to the `front` directory:**
   ```bash
   cd front
   ```

2. **Build the Docker image for the frontend:**
   ```bash
   docker build -t proyecto_aml_front .
   ```

3. **Run the Docker container for the frontend:**
   ```bash
   docker run -p 3000:3000 proyecto_aml_front
   ```

This will start both the backend service and the frontend inside Docker containers. Ensure the frontend is configured to communicate with the backend via the appropriate API endpoints.

By following these steps, you should have the entire project up and running, allowing you to interact with the backend API, use the frontend interface, and train the neural network model.

#### Model Training
The chosen model for deployment is DenseNet121 from Torch library. To train the model follow the next steps:
1. **Navigate to the `training` directory:**
   ```bash
   cd training
   ```
2. **Enter to the dataset folder where CheXPert dataset will be located**
   ```bash
   cd CheXpert-v1.0-small
   ```
3. **Download the CheXPert database from the following [link](https://uniandes-my.sharepoint.com/:u:/g/personal/s_rodriguez47_uniandes_edu_co/EXXVAEYAuIVFhGlSGKw0zVsB0BfaCdLAFEcdXlDLjp0IAw?e=5uoDw2)**
4. **Unzip the folder to obtain the dataset**
    ```bash
    unzip CheXpert-v1.0-small.zip
    ```
6. **Locate yourself at the level of the folder training**
    ```bash
    cd ..
    ```
7. **Run the main3.py**
    ```bash
    python main3.py
    ```
