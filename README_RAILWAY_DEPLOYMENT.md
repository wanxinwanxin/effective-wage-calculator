# Railway Deployment Guide

## 🚀 Deployment Setup

### 1. Required Environment Variables
Set these in your Railway project dashboard:

```
OPENAI_API_KEY=your_actual_openai_api_key
```

### 2. Deployment Configuration
The following files have been configured for Railway:

- **`railway.toml`**: Main Railway configuration
- **`Procfile`**: Alternative startup command specification  
- **`runtime.txt`**: Python version specification
- **`backend/requirements.txt`**: Python dependencies

### 3. Project Structure
```
effective-wage-calculator/
├── backend/
│   ├── main.py          # FastAPI application
│   ├── requirements.txt # Python dependencies
│   └── .env.example     # Environment variables template
├── railway.toml         # Railway configuration
├── Procfile            # Process file
└── runtime.txt         # Python version
```

## 🔧 Troubleshooting Common Issues

### Issue 1: "Module not found" errors
**Solution**: Ensure all dependencies are in `backend/requirements.txt`

### Issue 2: App won't start / Port binding errors
**Solution**: Railway automatically provides `$PORT` environment variable. The app is configured to use it.

### Issue 3: "No such file or directory" errors
**Solution**: The startup command includes `cd backend` to ensure proper working directory.

### Issue 4: OpenAI API errors
**Solution**: Set `OPENAI_API_KEY` in Railway dashboard. The app will fallback to basic calculations if the API key is missing.

### Issue 5: CORS errors from frontend
**Solution**: The backend allows all origins (`allow_origins=["*"]`). Update this in production for security.

## 📋 Deployment Checklist

- [ ] Set `OPENAI_API_KEY` in Railway environment variables
- [ ] Verify Python version (3.11) is supported
- [ ] Check Railway build logs for dependency installation
- [ ] Test health check endpoint: `GET /`
- [ ] Verify main API endpoint: `POST /calculate`

## 🔗 API Endpoints

- **Health Check**: `GET /` 
- **Calculate EHW**: `POST /calculate`
- **Research Facts**: `GET /research-facts`
- **Classification Info**: `GET /classification-info`

## 🚨 If Deployment Still Fails

1. Check Railway build logs for specific error messages
2. Verify all files are committed to your repository
3. Ensure the Railway service is pointing to the correct repository/branch
4. Try redeploying with the "Deploy" button in Railway dashboard

## 📝 Local Development

To run locally:
```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your OpenAI API key
uvicorn main:app --reload --port 8000
``` 