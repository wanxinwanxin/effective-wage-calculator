# Railway detection helper - imports the actual FastAPI app from backend
import sys
import os
sys.path.append('backend')

from backend.main import app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 