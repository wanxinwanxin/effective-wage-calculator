# Effective Hourly Wage (EHW) Calculator - MVP

A simple web application that calculates the economic value of any activity you do, expressed as an effective hourly wage.

## What It Does

Enter any activity (like "went for a run", "scrolled social media", or "learned programming") along with basic info about yourself, and get:

- **Effective Hourly Wage**: The net economic value of that activity per hour
- **Activity Classification**: Automatically categorizes your activity 
- **Detailed Explanation**: Step-by-step breakdown of how the value was calculated

## Quick Start

### 1. Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Start the API Server

```bash
cd backend
python main.py
```

The API will run at `http://localhost:8000`

### 3. Open the Web Interface

Simply open `frontend/index.html` in your web browser.

## Try These Examples

1. **Positive EHW**: "Went for a 45-minute jog" → Should show positive value due to health benefits
2. **Negative EHW**: "Scrolled Instagram for 2 hours" → Should show negative value due to opportunity cost
3. **Learning**: "Studied Python programming for 1 hour" → Should show positive value due to skill development

## How It Works (MVP Version)

This is a simplified version with:

- **3 Activity Archetypes**: Cardio exercise, social media browsing, skill learning
- **Deterministic Calculation**: No uncertainty bands yet (coming in future versions)
- **Hardcoded Parameters**: Based on research averages (will be data-driven later)

### Current Calculation Logic

**Cardio Exercise**: Health benefits + mood boost - opportunity cost
**Social Media**: -(Productivity loss + stress cost + opportunity cost)  
**Skill Learning**: Future earning potential + satisfaction - opportunity cost

## Architecture

```
Frontend (HTML/JS) → FastAPI Backend → Calculation Engine → Results
```

## Next Steps

This MVP demonstrates the core concept. The full system (from `Implementation_Instructions.md`) will add:

- Uncertainty quantification with confidence bands
- Research-backed fact database  
- DAG-based calculation engine
- More activity archetypes
- Personalization based on user feedback

## API Endpoints

- `GET /` - Health check
- `POST /calculate` - Calculate EHW for an activity
- `GET /archetypes` - List available activity types

## File Structure

```
├── backend/
│   ├── main.py              # FastAPI application
│   └── requirements.txt     # Python dependencies
├── frontend/
│   └── index.html          # Web interface
└── README.md
``` 