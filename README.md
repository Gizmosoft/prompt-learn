# PromptLearn: A Comprehensive Prompt Performance Analyzer

## Overview

**PromptLearn** is a full-stack application designed to help developers and researchers systematically test, compare, and optimize Large Language Model (LLM) prompts. The system enables users to run multiple prompt variants against the same input data, measure performance metrics, track costs, and visualize results through an intuitive web interface.

## Problem Statement

When developing LLM applications, choosing the right prompt can significantly impact:
- **Response Quality**: Different phrasings yield different results
- **Cost Efficiency**: Token usage varies dramatically between prompts
- **Latency**: Some prompts generate faster responses than others
- **Reliability**: Understanding which prompts work consistently

Manually testing and comparing prompts is time-consuming and error-prone. PromptLearn automates this process, providing quantitative insights to make data-driven decisions.

## Architecture

The project follows a clean separation of concerns with three main layers:

### 1. **Backend API (FastAPI)**
This RESTful API serves as the core service layer:

- **Technology Stack**: FastAPI, Pydantic for request validation, SQLite for persistence
- **Key Endpoints**:
  - `POST /api/experiments/run`: Execute a new experiment with multiple prompt variants
  - `GET /api/experiments`: List all experiments
  - `GET /api/experiments/{experiment_id}`: Retrieve detailed results for a specific experiment

- **Features**:
  - Asynchronous request handling to prevent blocking during long-running experiments
  - CORS middleware for frontend integration
  - Structured error handling with HTTP status codes
  - Type-safe request/response models using Pydantic

### 2. **Prompt Runner Service**
This is the heart of the system with core functionality.

**Core Responsibilities**:
- Orchestrates multi-variant prompt execution
- Integrates with Google Gemini models via LangChain
- Tracks performance metrics (latency, tokens, cost)
- Persists all data to SQLite database
- Optional Langfuse integration for advanced tracing

**Key Components**:

1. **Experiment Execution**:
   - Accepts configuration with multiple prompt variants
   - Runs each variant sequentially against the same input
   - Captures timing, token usage, and response data
   - Handles errors gracefully without stopping the entire experiment

2. **Cost Calculation**:
   - Automatic cost estimation based on token usage
   - Supports custom pricing for different models

3. **Metrics Extraction**:
   - Extracts token counts from Gemini API responses
   - Handles multiple response metadata formats
   - Provides confidence levels ("measured" vs "unknown") for metrics

4. **Langfuse Integration**:
   - Optional observability platform integration
   - Generates trace URLs for each run
   - Enables detailed debugging and performance analysis
   - Gracefully degrades if credentials are missing

### 3. **Frontend Dashboard (Streamlit)**
An interactive web interface for experiment management:

**Features**:
- **Experiment Creation**: 
  - Form-based interface for defining experiments
  - Dynamic variant management (add/remove prompt templates)
  - Input text configuration with variable substitution
  - Model and temperature selection

- **Results Visualization**:
  - Summary metrics dashboard (total variants, success rate, average latency, total cost)
  - Comparative charts using Plotly:
    - Latency comparison across variants
    - Cost comparison
    - Token usage breakdown (input vs output)
  - Detailed variant information with expandable sections

- **Data Management**:
  - Experiment listing with search and filtering
  - Historical experiment tracking
  - Real-time data refresh capabilities

**UI/UX Enhancements**:
- Custom CSS styling for consistent branding
- Responsive layout with column-based organization
- Color-coded status indicators (success, error, running)
- Interactive charts with hover tooltips

## Database Schema
The SQLite database uses a normalized schema:

1. **experiments**: Stores experiment metadata (ID, name, model, temperature, creation timestamp)
2. **prompt_variants**: Defines prompt templates for each experiment (label, template text, optional notes)
3. **runs**: Tracks individual execution instances (status, timestamps, error messages)
4. **run_metrics**: Performance data (latency, tokens, cost, trace URLs, confidence levels)
5. **run_outputs**: LLM responses (text and raw JSON)

**Design Decisions**:
- UUID-based primary keys for distributed system compatibility
- Input hash for idempotency tracking
- Foreign key relationships for data integrity
- WAL (Write-Ahead Logging) mode for better concurrency

## Key Workflows

### Creating an Experiment

1. User fills out the experiment form:
   - Provides experiment name (optional)
   - Selects model (Gemini Flash or Pro)
   - Sets temperature parameter
   - Enters input text that will be used across all variants
   - Defines one or more prompt templates using `{text}` variable

2. Frontend sends POST request to `/api/experiments/run`

3. Backend processes the request:
   - Creates experiment record in database
   - For each variant:
     - Formats template with input variables
     - Executes LLM call via LangChain
     - Measures latency
     - Extracts token usage
     - Calculates cost
     - Persists all metrics and outputs
   - Returns aggregated results

4. Frontend displays results with visualizations

### Viewing Results

1. User selects an experiment from the list
2. Frontend fetches experiment details via `GET /api/experiments/{id}`
3. Backend queries database with JOINs to aggregate:
   - Experiment metadata
   - All variant results
   - Metrics and outputs
4. Frontend renders:
   - Summary statistics
   - Comparative charts
   - Detailed variant breakdowns

## Technical Highlights

### Error Handling
- Graceful degradation if Langfuse credentials are missing
- Individual variant failures don't stop the entire experiment
- Detailed error messages stored in database for debugging
- Frontend displays errors with clear user feedback

### Performance Optimization
- Asynchronous API endpoints prevent blocking
- Database connection pooling via context managers
- Efficient SQL queries with proper indexing
- Caching in frontend for frequently accessed data

### Extensibility
- Modular service architecture allows easy model swapping
- Pricing tables can be extended for new models
- Database schema supports additional metrics
- Frontend components are reusable and configurable

## Technology Stack

**Backend**:
- FastAPI: Modern, fast web framework
- LangChain: LLM orchestration and abstraction
- Langfuse: Observability and tracing (optional)
- SQLite: Lightweight, embedded database
- Pydantic: Data validation and serialization

**Frontend**:
- Streamlit: Rapid web app development
- Plotly: Interactive data visualization
- Pandas: Data manipulation for charts

**Infrastructure**:
- Python 3.x
- Environment-based configuration (.env files)
- CORS-enabled for cross-origin requests

## Use Cases

1. **Prompt Engineering**: Systematically test different prompt phrasings to find optimal wording
2. **Cost Optimization**: Compare token usage and costs across variants to minimize expenses
3. **Performance Tuning**: Identify which prompts generate faster responses
4. **A/B Testing**: Run controlled experiments with multiple prompt strategies
5. **Quality Assurance**: Track consistency and reliability of different prompts over time

## Local Project Setup  

1. Create a .env file based on the .env.text file  
2. To run backend server, `cd backend` and run the command `uvicorn backend.api.main:app --reload`  
3. Then `cd frontend` and run `streamlit run dashboard.py` to start the frontend app  

## Conclusion

PromptLearn provides a comprehensive solution for prompt optimization, combining the power of modern LLM frameworks with practical performance tracking and visualization. By automating the tedious process of prompt comparison, it enables developers to make informed decisions based on quantitative data rather than intuition.

The system's modular architecture, robust error handling, and intuitive interface make it suitable for both individual developers and teams working on LLM-powered applications. Whether you're optimizing for cost, speed, or quality, PromptLearn gives you the insights needed to choose the best prompt for your use case.

