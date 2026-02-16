# B3 (Bit-By-Bit)

A full-stack AI-powered data science workspace that enables dataset management, exploratory analysis, intelligent planning, model training, hyperparameter tuning, evaluation, and conversational orchestration through a unified web interface.

This system combines a FastAPI backend, a React frontend, and an Ollama-powered planning agent to create an interactive machine learning research environment.

---

# Overview

The AI Research Workspace provides:

- CSV dataset upload and registration
- Dataset preview and profiling
- Conversational AI planning
- Multi-model training (regression and classification)
- Hyperparameter tuning
- Model evaluation
- Prediction API
- Real-time inspector panel
- Interactive web console

The system supports a CLI-style experience through a modern web interface.

---

# Architecture

## Backend (FastAPI)

The backend is modular and organized into the following components:

### Core Systems

- Orchestrator
- Dispatcher
- ChatController
- Planner
- PlanExecutor

### Engines

- DataEngine
- EDAEngine
- DataProfiler
- MLEngine
- EvaluationEngine
- ExplainEngine
- DataStrategistEngine

### State Management

- DatasetRegistry
- ModelRegistry
- Centralized session memory

### API Endpoints

- `POST /upload-file`
- `GET /datasets`
- `GET /preview/{dataset_id}`
- `GET /profile/{dataset_id}`
- `POST /set-active-dataset`
- `POST /chat`
- `GET /inspector`

---

## Frontend (React + Tailwind)

The frontend includes:

- Animated gradient workspace
- Sidebar navigation
- Dataset modal panel
- Dataset preview tab
- Column profiling tab
- Integrated chat panel
- Collapsible console
- Real-time inspector

Main layout areas:

- Sidebar
- Workspace
- Inspector panel
- Console panel

---

# Features

## Dataset Management

- Upload CSV datasets
- Automatic in-memory registration
- List available datasets
- Select active dataset
- Preview dataset rows
- View column statistics

---

## Conversational Planning

The system uses an Ollama model to:

- Interpret user intent
- Generate structured execution plans
- Inject required parameters
- Track dataset and model context
- Require confirmation before execution

---

## Data Profiling

The DataProfiler engine provides:

- Dataset summary
- Column data types
- Missing value analysis
- Numeric statistics
- High cardinality detection
- Multicollinearity detection
- Target candidate inference

---

## Model Training

Supported regression models:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

Supported classification models:

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

Features:

- Automatic train-test split
- Metadata tracking
- Feature validation
- Model ID registration

---

## Hyperparameter Tuning

Uses GridSearchCV to tune:

- Random Forest
- Gradient Boosting

Stores best parameters and tuned model metadata.

---

## Model Evaluation

Regression metrics:

- R2
- MAE
- RMSE

Classification metrics:

- Accuracy
- Precision
- Recall
- F1 Score

---

## Prediction

- Validates required features
- Preserves feature order
- Returns structured prediction response

---

## Inspector Panel

Displays:

- Active dataset
- Dataset size
- Active model
- Problem type
- Target column
- Model type
- Metadata

Automatically updates after dataset selection and model training.

---

## Console System

- Scrollable
- Collapsible
- Real-time execution logs
- Chat transcripts
- Tool responses

---

# End-to-End Flow

1. Upload dataset
2. Select active dataset
3. Preview and profile
4. Define problem type and target
5. Confirm training plan
6. Train model
7. Evaluate model
8. Tune model if required
9. Generate predictions

---

# Running the Backend

```bash
uvicorn app.main:app --reload 

```

```bash
cd frontend
npm start
```

# Technology Stack

## Backend:

FastAPI

Pandas

Scikit-learn

GridSearchCV

Ollama

Pydantic

## Frontend:

React

Tailwind CSS

Framer Motion

Axios
