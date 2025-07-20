#!/bin/bash

echo "🚀 Effective Hourly Wage Calculator - Deployment Script"
echo "===================================================="

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Installing..."
    npm install -g @railway/cli
fi

# Check if logged in to Railway
if ! railway whoami &> /dev/null; then
    echo "🔑 Please log in to Railway:"
    railway login
fi

echo ""
echo "📦 Deploying backend to Railway..."

# Deploy to Railway
railway new --name "effective-wage-calculator"
railway add
railway up

echo ""
echo "⚙️ Setting up environment variables..."
echo "Please add your OpenAI API key (optional but recommended):"
read -p "OpenAI API Key (press enter to skip): " api_key

if [ -n "$api_key" ]; then
    railway variables set OPENAI_API_KEY="$api_key"
    echo "✅ API key added"
else
    echo "⚠️ Skipping API key - heuristic fallback will be used"
fi

echo ""
echo "🌐 Getting deployment URL..."
sleep 5  # Wait for deployment to complete

# Get the Railway URL
RAILWAY_URL=$(railway status --json | grep -o '"url":"[^"]*' | cut -d'"' -f4)

if [ -n "$RAILWAY_URL" ]; then
    echo "✅ Backend deployed at: $RAILWAY_URL"
    
    echo ""
    echo "📝 Next steps:"
    echo "1. Update frontend API URL in:"
    echo "   - frontend/config.js"
    echo "   - ../wanxinwanxin.github.io/ehw-calculator.html"
    echo ""
    echo "2. Replace 'https://effective-wage-calculator-production.up.railway.app' with:"
    echo "   $RAILWAY_URL"
    echo ""
    echo "3. Commit and push your website changes:"
    echo "   cd ../wanxinwanxin.github.io"
    echo "   git add ."
    echo "   git commit -m 'Add EHW Calculator with live backend'"
    echo "   git push"
    echo ""
    echo "🎉 Your calculator will then be live at:"
    echo "   https://wanxinwanxin.github.io/ehw-calculator.html"
    
else
    echo "❌ Could not get Railway URL. Check deployment status:"
    echo "   railway status"
fi

echo ""
echo "🔧 Additional commands:"
echo "  railway logs     - View backend logs"
echo "  railway open     - Open Railway dashboard"
echo "  railway redeploy - Redeploy if needed" 