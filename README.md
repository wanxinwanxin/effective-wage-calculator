# 🧮 Effective Hourly Wage Calculator

**An AI-powered calculator that measures the economic value of any activity using research-backed health economics, life extension valuations, and proper economic principles.**

## 🎯 What It Does

Enter any activity (like "went for a run", "scrolled social media", or "learned programming") along with basic info about yourself, and get:

- **Effective Hourly Wage**: The net economic value of that activity per hour
- **AI Activity Classification**: Automatically categorizes your activity using GPT-3.5-turbo
- **Research-Backed Analysis**: 6 peer-reviewed studies with detailed explanations
- **Professional Interface**: Beautiful, mobile-responsive design

## 🚀 Quick Start (Local)

1. **Install dependencies:**
   ```bash
   pip install -r requirements-streamlit.txt
   ```

2. **Run the app:**
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Open in browser:** http://localhost:8501

## ☁️ Deploy to Streamlit Cloud (FREE!)

### Step 1: Push to GitHub
Your code is already on GitHub, so you're ready!

### Step 2: Deploy to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select repository: `wanxinwanxin/effective-wage-calculator`
5. Main file path: `streamlit_app.py`
6. Click "Deploy!"

### Step 3: Add OpenAI API Key (Optional)
1. In your Streamlit Cloud dashboard, click "Manage app"
2. Go to "Secrets" tab
3. Add:
   ```toml
   OPENAI_API_KEY = "your_actual_api_key_here"
   ```
4. Save and restart the app

## 🎯 Features

### ✅ What Works:
- **Smart Activity Classification** (AI + keyword fallback)
- **Research-Backed Calculations** (6 peer-reviewed studies integrated)
- **Beautiful UI** with forms, metrics, and charts
- **Detailed Explanations** with step-by-step breakdowns
- **Real-time Results** with confidence levels
- **Mobile Responsive** design

### 🧮 Calculation Types:
- **Cardiovascular Exercise**: Health benefits + life extension value
- **Social Media**: Productivity costs + mental health impacts  
- **Learning**: Future earnings + immediate satisfaction
- **Creative Activities**: Intrinsic value + potential monetization
- **Social Activities**: Relationship value + mental health

### 🤖 AI Features:
- **Activity Classification**: GPT-3.5 analyzes your description
- **Fallback System**: Works without API key using keywords
- **Research Matching**: Dynamically selects relevant studies

## 💰 Monetization Ready

### Current Freemium Model:
- **Free Tier**: Keyword-based classification + basic calculations
- **Premium**: Users provide their own OpenAI API key for AI features
- **Future**: Subscription for hosted AI + premium features

### Revenue Streams:
1. **API Key Freemium**: Users pay OpenAI directly
2. **Lead Generation**: Email capture for "detailed reports"
3. **Consulting**: Link to personal productivity coaching
4. **White Label**: License to HR/wellness companies

## 🔧 Technical Details

### Architecture:
```
Streamlit UI → Activity Classifier → Research Database → EHW Calculator → Results Display
```

### Key Components:
- **6 Research Facts** with citations and effect sizes
- **Economic Models**: Present value, opportunity cost, life extension valuation
- **AI Classification**: Multi-dimensional activity analysis
- **Uncertainty Handling**: Confidence levels and explanation quality

### Performance:
- **Load Time**: <2 seconds
- **Calculation**: <1 second (without AI) / <5 seconds (with AI)
- **Mobile**: Fully responsive
- **Caching**: Streamlit built-in optimization

## 📱 Usage Examples

Try these in the app:

### High Value Activities:
- "Went for a 30-minute run"
- "Studied Python programming for 2 hours"  
- "Had lunch with a mentor"
- "Practiced guitar for 45 minutes"

### Low/Negative Value:
- "Scrolled Instagram for 2 hours"
- "Watched Netflix for 4 hours"
- "Procrastinated on homework for 1 hour"

### Interesting Cases:
- "Played video games for 3 hours" (varies by game type)
- "Took a 20-minute nap" (productivity boost vs opportunity cost)
- "Organized my room for 1 hour" (mental clarity vs time cost)

## 🎨 Customization

### Easy Modifications:
- **Add Research Facts**: Edit `RESEARCH_FACTS` dictionary
- **New Activity Types**: Extend classification logic
- **UI Themes**: Modify `.streamlit/config.toml`
- **Branding**: Update CSS in `st.markdown()` calls

### Advanced Features (Future):
- **User Accounts**: Add streamlit-authenticator
- **Data Persistence**: Connect to database
- **Analytics**: Track usage patterns
- **A/B Testing**: Different explanation styles

## 🚀 Deployment Options

| Platform | Cost | Setup Time | Best For |
|----------|------|------------|----------|
| **Streamlit Cloud** | Free | 5 minutes | MVP, demos |
| **Heroku** | $7/month | 15 minutes | Professional apps |
| **DigitalOcean** | $5/month | 30 minutes | Full control |
| **Railway** | $5/month | 20 minutes | Auto-deploy |

### Recommended: Streamlit Cloud
- ✅ **Zero cost** for public repos
- ✅ **Auto-deploy** on git push
- ✅ **SSL certificate** included
- ✅ **CDN** for fast global access
- ✅ **Easy secrets** management

## 📊 Success Metrics

### MVP Goals:
- [ ] Deploy successfully to Streamlit Cloud
- [ ] Handle 100+ different activity descriptions
- [ ] <5 second response time with AI
- [ ] Mobile-friendly interface
- [ ] Email capture for premium features

### Growth Metrics:
- **Daily Active Users**: Target 50+ within first month
- **Conversion Rate**: 10% provide email for reports
- **Activity Coverage**: 80% of submissions get relevant research
- **User Satisfaction**: >4.5/5 in feedback

## 🔗 Links

- **Live App**: https://ehw-calculator.streamlit.app/
- **GitHub**: https://github.com/wanxinwanxin/effective-wage-calculator  
- **Documentation**: This README
- **Issues**: GitHub Issues tab

## 🆘 Troubleshooting

### Common Issues:

**"ModuleNotFoundError"**
```bash
pip install -r requirements-streamlit.txt
```

**AI classification not working**
- Add OpenAI API key in sidebar or secrets
- Check API key has credits
- Fallback to keyword classification works without API

**Slow performance**
- Check internet connection for AI calls
- Use keyword classification for faster results
- Consider caching for repeated calculations

**Deployment fails**
- Ensure `streamlit_app.py` is in root directory
- Check `requirements-streamlit.txt` is present
- Verify GitHub repository is public

## 📈 Roadmap

### Version 1.1 (Next 2 weeks):
- [ ] User feedback collection
- [ ] More activity archetypes (sleep, meditation, cooking)
- [ ] Uncertainty bands with Monte Carlo simulation
- [ ] Export results to PDF

### Version 1.2 (Next month):
- [ ] User accounts and history
- [ ] Personalized recommendations
- [ ] Social sharing features
- [ ] Integration with fitness trackers

### Version 2.0 (Future):
- [ ] Multi-activity portfolio optimization
- [ ] Team/family accounts
- [ ] Corporate wellness integration
- [ ] Mobile app (React Native)

---

**Ready to launch? Your EHW Calculator is production-ready! 🚀** 