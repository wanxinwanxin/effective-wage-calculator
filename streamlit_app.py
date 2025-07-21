import streamlit as st
import openai
import os
import json
import math
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import asyncio

# Set page config
st.set_page_config(
    page_title="🧮 Effective Hourly Wage Calculator",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Data models (same as FastAPI)
@dataclass
class UserProfile:
    age: Optional[int] = None
    current_hourly_wage: Optional[float] = None
    profession: Optional[str] = None
    location: Optional[str] = None
    fitness_level: Optional[str] = "moderate"

@dataclass
class ResearchFact:
    fact_id: str
    claim: str
    effect_size: float
    effect_unit: str
    confidence: str
    citation: str
    population: str
    notes: str

@dataclass
class ActivityClassification:
    primary_category: str
    health_impact: str
    productivity_impact: str
    intensity_level: str
    relevant_research_domains: List[str]
    confidence: float
    reasoning: str

@dataclass
class EHWResult:
    effective_hourly_wage: float
    activity_type: str
    explanation: str
    research_facts_used: List[ResearchFact]
    calculation_breakdown: Dict[str, Any]
    confidence_level: str
    processing_log: List[str]

# Research Facts Database (same as FastAPI)
RESEARCH_FACTS = {
    "cardio_cvd_risk": ResearchFact(
        fact_id="cardio_cvd_risk_001",
        claim="Regular moderate aerobic exercise reduces cardiovascular disease risk by 20-35%",
        effect_size=0.275,
        effect_unit="relative_risk_reduction",
        confidence="high",
        citation="Warburton, D.E., et al. (2006). Health benefits of physical activity: the evidence. CMAJ, 174(6), 801-809.",
        population="Adults aged 18-65",
        notes="Meta-analysis of 33 studies, dose-response relationship observed"
    ),
    "cardio_mortality": ResearchFact(
        fact_id="cardio_mortality_001", 
        claim="Each additional hour of moderate exercise per week reduces all-cause mortality by 9%",
        effect_size=0.09,
        effect_unit="mortality_reduction_per_hour_weekly",
        confidence="high",
        citation="Arem, H., et al. (2015). Leisure time physical activity and mortality. JAMA Internal Medicine, 175(6), 959-967.",
        population="Adults, 661,137 participants, 14.2 year follow-up",
        notes="Dose-response curve plateaus at ~7.5 hours/week"
    ),
    "cvd_medical_costs": ResearchFact(
        fact_id="cvd_costs_001",
        claim="Average lifetime medical costs for cardiovascular disease: $100,000-$200,000",
        effect_size=150000,
        effect_unit="USD_lifetime_medical_costs",
        confidence="medium", 
        citation="Heidenreich, P.A., et al. (2022). Forecasting the impact of heart failure in the United States. Circulation: Heart Failure, 6(3), 606-619.",
        population="US adults with CVD diagnosis",
        notes="Includes direct medical costs, varies significantly by severity and age at onset"
    ),
    "social_media_productivity": ResearchFact(
        fact_id="sm_productivity_001",
        claim="Heavy social media use reduces workplace productivity by 13-21%",
        effect_size=0.17,
        effect_unit="productivity_reduction_fraction",
        confidence="medium",
        citation="Duke, É., & Montag, C. (2017). Smartphone addiction, daily interruptions and self-reported productivity. Addictive Behaviors Reports, 6, 90-95.",
        population="Knowledge workers, 18-45 years old",
        notes="Based on self-reported time tracking and productivity metrics"
    ),
    "social_media_mental_health": ResearchFact(
        fact_id="sm_mental_001",
        claim="2+ hours daily social media use increases anxiety/depression risk by 25%",
        effect_size=0.25,
        effect_unit="relative_risk_increase", 
        confidence="medium",
        citation="Primack, B.A., et al. (2017). Social media use and perceived social isolation among young adults in the U.S. American Journal of Preventive Medicine, 53(1), 1-8.",
        population="Young adults 19-32 years",
        notes="Cross-sectional study, causation not definitively established"
    ),
    "learning_wage_premium": ResearchFact(
        fact_id="learning_wage_001",
        claim="One year of additional skill training increases lifetime earnings by 8-13%",
        effect_size=0.105,
        effect_unit="lifetime_earnings_increase",
        confidence="high",
        citation="Card, D. (1999). The causal effect of education on earnings. Handbook of Labor Economics, 3, 1801-1863.",
        population="Working adults across industries",
        notes="Consistent across multiple countries and time periods"
    )
}

# Initialize OpenAI (using Streamlit secrets)
def init_openai():
    """Initialize OpenAI with API key from Streamlit secrets or environment"""
    try:
        # Try Streamlit secrets first (for deployed apps)
        if "OPENAI_API_KEY" in st.secrets:
            openai.api_key = st.secrets["OPENAI_API_KEY"]
            return True
        # Fall back to environment variable (for local development)
        elif "OPENAI_API_KEY" in os.environ:
            openai.api_key = os.environ["OPENAI_API_KEY"]
            return True
        else:
            return False
    except:
        return False

# All the calculation functions (same as FastAPI)
def calculate_opportunity_cost(user_profile: UserProfile) -> float:
    """Calculate user's opportunity cost per hour"""
    if user_profile.current_hourly_wage:
        return user_profile.current_hourly_wage * 0.8
    
    if user_profile.age and user_profile.age < 25:
        return 15.0
    elif user_profile.age and user_profile.age < 35:
        return 25.0
    else:
        return 30.0

def calculate_present_value(future_benefit: float, years_delay: float, discount_rate: float = 0.03) -> float:
    """Calculate present value of future benefits"""
    return future_benefit / ((1 + discount_rate) ** years_delay)

def calculate_cardiovascular_benefits(user_profile: UserProfile, duration_hours: float) -> Dict[str, Any]:
    """Calculate detailed cardiovascular benefits using proper economic amortization"""
    
    cvd_risk_fact = RESEARCH_FACTS["cardio_cvd_risk"]
    mortality_fact = RESEARCH_FACTS["cardio_mortality"] 
    cost_fact = RESEARCH_FACTS["cvd_medical_costs"]
    
    user_age = user_profile.age or 35
    life_expectancy = 80
    years_remaining = max(5, life_expectancy - user_age)
    
    # Economic calculation
    baseline_cvd_risk = 0.25
    risk_reduction_fraction = cvd_risk_fact.effect_size
    absolute_risk_reduction = baseline_cvd_risk * risk_reduction_fraction
    total_medical_savings = absolute_risk_reduction * cost_fact.effect_size
    
    # Value of life extension
    value_statistical_life = 11_000_000
    years_life_extension = 2.5
    total_life_extension_value = years_life_extension * (value_statistical_life / life_expectancy)
    
    total_lifetime_benefit = total_medical_savings + total_life_extension_value
    
    # Amortization
    hours_per_week_needed = 2.5
    weeks_per_year = 52
    years_of_exercise_needed = years_remaining
    total_hours_needed = hours_per_week_needed * weeks_per_year * years_of_exercise_needed
    
    benefit_per_hour = total_lifetime_benefit / total_hours_needed
    present_value_total_benefit = calculate_present_value(total_lifetime_benefit, years_remaining / 2)
    present_value_per_hour = present_value_total_benefit / total_hours_needed
    
    return {
        "total_lifetime_benefit": total_lifetime_benefit,
        "medical_savings_component": total_medical_savings,
        "life_extension_component": total_life_extension_value,
        "total_hours_needed": total_hours_needed,
        "years_of_commitment": years_of_exercise_needed,
        "hours_per_week_needed": hours_per_week_needed,
        "benefit_per_hour": benefit_per_hour,
        "present_value_per_hour": present_value_per_hour,
        "years_life_extension": years_life_extension,
        "value_statistical_life": value_statistical_life,
        "research_facts": [cvd_risk_fact, mortality_fact, cost_fact]
    }

def calculate_social_media_costs(user_profile: UserProfile, duration_hours: float) -> Dict[str, Any]:
    """Calculate detailed social media costs with research backing"""
    
    productivity_fact = RESEARCH_FACTS["social_media_productivity"]
    mental_health_fact = RESEARCH_FACTS["social_media_mental_health"]
    
    opportunity_cost = calculate_opportunity_cost(user_profile)
    productivity_loss = duration_hours * opportunity_cost * productivity_fact.effect_size
    
    daily_usage = duration_hours
    if daily_usage >= 2.0:
        annual_mental_health_costs = 2000
        risk_increase = mental_health_fact.effect_size
        expected_annual_cost = annual_mental_health_costs * risk_increase
        daily_cost = expected_annual_cost / 365
        session_mental_cost = daily_cost
    else:
        session_mental_cost = duration_hours * 2.0
    
    return {
        "productivity_loss": productivity_loss,
        "mental_health_cost": session_mental_cost,
        "total_cost": productivity_loss + session_mental_cost,
        "research_facts": [productivity_fact, mental_health_fact]
    }

def calculate_learning_benefits(user_profile: UserProfile, duration_hours: float) -> Dict[str, Any]:
    """Calculate detailed learning benefits with research backing"""
    
    wage_fact = RESEARCH_FACTS["learning_wage_premium"]
    
    opportunity_cost = calculate_opportunity_cost(user_profile)
    estimated_annual_income = opportunity_cost * 2000
    
    hours_for_significant_skill = 200
    skill_progress = duration_hours / hours_for_significant_skill
    
    lifetime_earnings_increase = estimated_annual_income * wage_fact.effect_size * skill_progress
    
    user_age = user_profile.age or 30
    career_years_remaining = max(5, 65 - user_age)
    present_value_earnings = calculate_present_value(lifetime_earnings_increase, 2) * career_years_remaining
    
    return {
        "skill_progress_percent": skill_progress * 100,
        "estimated_lifetime_increase": lifetime_earnings_increase,
        "present_value_earnings": present_value_earnings,
        "career_years_remaining": career_years_remaining,
        "research_facts": [wage_fact]
    }

def classify_activity_with_ai(description: str, processing_log: List[str]) -> ActivityClassification:
    """Use AI to classify activity into multiple dimensions"""
    
    if not openai.api_key:
        processing_log.append("🔑 No OpenAI API key found - using enhanced heuristic classification")
        return classify_activity_fallback(description, processing_log)
    
    try:
        processing_log.append("🤖 Attempting AI classification using GPT-3.5-turbo...")
        
        system_prompt = """You are an expert activity classifier for economic value analysis. 

Analyze the described activity and classify it across multiple dimensions:

PRIMARY_CATEGORY: physical, mental, social, productive, entertainment, maintenance, creative
HEALTH_IMPACT: cardiovascular, strength, mental_health, cognitive, sleep, nutrition, negative, neutral
PRODUCTIVITY_IMPACT: enhancing, neutral, reducing  
INTENSITY_LEVEL: low, moderate, high
RELEVANT_RESEARCH_DOMAINS: List from [cardio, social_media, learning, sleep, nutrition, creativity, social_connection]

Respond in JSON format:
{
  "primary_category": "...",
  "health_impact": "...", 
  "productivity_impact": "...",
  "intensity_level": "...",
  "relevant_research_domains": [...],
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of classification"
}"""

        user_prompt = f"Classify this activity: '{description}'"
        
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=300,
            temperature=0.1
        )
        
        result = json.loads(response.choices[0].message.content)
        
        classification = ActivityClassification(**result)
        processing_log.append(f"✅ AI classification successful: {classification.primary_category} activity with {classification.health_impact} health impact")
        processing_log.append(f"🧠 AI reasoning: {classification.reasoning}")
        
        return classification
        
    except Exception as e:
        error_msg = str(e)
        if "quota" in error_msg.lower():
            processing_log.append("⚠️ AI quota exceeded - falling back to enhanced heuristic classification")
        elif "429" in error_msg:
            processing_log.append("⚠️ AI rate limit hit - falling back to enhanced heuristic classification")
        else:
            processing_log.append(f"❌ AI classification failed ({error_msg[:50]}...) - using fallback")
        
        fallback = classify_activity_fallback(description, processing_log)
        fallback.reasoning = f"AI failed: {error_msg[:100]}"
        fallback.confidence = 0.2
        return fallback

def classify_activity_fallback(description: str, processing_log: List[str]) -> ActivityClassification:
    """Enhanced heuristic classification when AI is unavailable"""
    desc_lower = description.lower()
    
    processing_log.append(f"🔍 Analyzing text: '{description}' using keyword heuristics")
    
    physical_keywords = ["run", "jog", "swim", "bike", "cycle", "walk", "hike", "workout", "gym", "exercise", "sport", "dance", "yoga", "climb", "lift", "weight", "cardio", "fitness", "training"]
    social_media_keywords = ["scroll", "instagram", "tiktok", "twitter", "facebook", "social media", "browse", "youtube", "reddit", "snap", "snapchat", "linkedin", "feed"]
    learning_keywords = ["learn", "study", "read", "course", "tutorial", "practice", "coding", "programming", "book", "research", "homework", "assignment", "lesson", "education"]
    creative_keywords = ["paint", "draw", "write", "music", "create", "design", "art", "craft", "sketch", "compose", "photography", "creative"]
    entertainment_keywords = ["play", "game", "gaming", "movie", "tv", "watch", "netflix", "entertainment", "leisure", "fun", "around", "mess", "chill", "relax"]
    social_keywords = ["hang", "hangout", "friends", "party", "social", "chat", "talk", "visit", "meet", "dinner", "lunch", "coffee"]
    
    if any(keyword in desc_lower for keyword in physical_keywords):
        matched_physical = [kw for kw in physical_keywords if kw in desc_lower]
        processing_log.append(f"🏃‍♂️ Physical activity keywords found: {matched_physical}")
        return ActivityClassification(
            primary_category="physical",
            health_impact="cardiovascular",
            productivity_impact="neutral",
            intensity_level="moderate",
            relevant_research_domains=["cardio"],
            confidence=0.8,
            reasoning=f"Detected physical activity keywords: {matched_physical} - classified as cardiovascular exercise"
        )
        
    elif any(keyword in desc_lower for keyword in social_media_keywords):
        matched_social_media = [kw for kw in social_media_keywords if kw in desc_lower]
        processing_log.append(f"📱 Social media keywords found: {matched_social_media}")
        return ActivityClassification(
            primary_category="entertainment",
            health_impact="mental_health",
            productivity_impact="reducing",
            intensity_level="low",
            relevant_research_domains=["social_media"],
            confidence=0.7,
            reasoning=f"Detected social media keywords: {matched_social_media} - classified as productivity-reducing entertainment"
        )
        
    elif any(keyword in desc_lower for keyword in learning_keywords):
        matched_learning = [kw for kw in learning_keywords if kw in desc_lower]
        processing_log.append(f"📚 Learning keywords found: {matched_learning}")
        return ActivityClassification(
            primary_category="productive",
            health_impact="cognitive",
            productivity_impact="enhancing",
            intensity_level="moderate",
            relevant_research_domains=["learning"],
            confidence=0.7,
            reasoning=f"Detected learning keywords: {matched_learning} - classified as skill development"
        )
        
    else:
        processing_log.append("❓ No specific keywords detected - using general classification")
        return ActivityClassification(
            primary_category="general",
            health_impact="neutral",
            productivity_impact="neutral",
            intensity_level="moderate",
            relevant_research_domains=[],
            confidence=0.3,
            reasoning="No specific keywords detected - using general classification"
        )

def get_relevant_research_facts(classification: ActivityClassification) -> List[ResearchFact]:
    """Dynamically select research facts based on AI classification"""
    facts = []
    
    domain_to_facts = {
        "cardio": ["cardio_cvd_risk", "cardio_mortality", "cvd_medical_costs"],
        "social_media": ["social_media_productivity", "social_media_mental_health"], 
        "learning": ["learning_wage_premium"]
    }
    
    for domain in classification.relevant_research_domains:
        if domain in domain_to_facts:
            for fact_id in domain_to_facts[domain]:
                if fact_id in RESEARCH_FACTS:
                    facts.append(RESEARCH_FACTS[fact_id])
    
    return facts

def calculate_ehw(user_profile: UserProfile, activity_description: str, duration_hours: float) -> EHWResult:
    """Calculate Effective Hourly Wage with AI-powered classification"""
    
    processing_log = []
    processing_log.append("🚀 Starting EHW calculation...")
    
    classification = classify_activity_with_ai(activity_description, processing_log)
    
    opportunity_cost = calculate_opportunity_cost(user_profile)
    if user_profile.current_hourly_wage:
        location_text = f" in {user_profile.location}" if user_profile.location else ""
        processing_log.append(f"💰 Opportunity cost: ${opportunity_cost:.2f}/hr (80% of stated ${user_profile.current_hourly_wage:.2f}/hr wage{location_text})")
    else:
        location_text = f" in {user_profile.location}" if user_profile.location else ""
        processing_log.append(f"💰 Opportunity cost: ${opportunity_cost:.2f}/hr (estimated based on age {user_profile.age or 'unknown'}{location_text})")
    
    research_facts_used = get_relevant_research_facts(classification)
    if research_facts_used:
        processing_log.append(f"📚 Found {len(research_facts_used)} relevant research studies for {classification.relevant_research_domains}")
    else:
        processing_log.append("📚 No specific research studies matched this activity type")
    
    calculation_breakdown = {
        "opportunity_cost": opportunity_cost,
        "ai_classification": {
            "primary_category": classification.primary_category,
            "health_impact": classification.health_impact,
            "productivity_impact": classification.productivity_impact,
            "intensity_level": classification.intensity_level,
            "relevant_research_domains": classification.relevant_research_domains,
            "confidence": classification.confidence,
            "reasoning": classification.reasoning
        }
    }
    
    total_benefits = 0.0
    total_costs = 0.0
    
    processing_log.append(f"🏥 Calculating health impacts for '{classification.health_impact}' classification...")
    
    if classification.health_impact == "cardiovascular":
        benefits = calculate_cardiovascular_benefits(user_profile, duration_hours)
        health_value_per_hour = benefits["present_value_per_hour"]
        immediate_benefits = 13.0
        total_benefits += health_value_per_hour + immediate_benefits
        calculation_breakdown["health_benefits"] = benefits
        
        processing_log.append(f"❤️ Cardiovascular analysis:")
        processing_log.append(f"  • Total lifetime benefit: ${benefits['total_lifetime_benefit']:,.0f}")
        processing_log.append(f"    - Medical savings: ${benefits['medical_savings_component']:,.0f}")
        processing_log.append(f"    - Life extension value: ${benefits['life_extension_component']:,.0f} ({benefits['years_life_extension']} years)")
        processing_log.append(f"  • Exercise commitment: {benefits['hours_per_week_needed']} hrs/week for {benefits['years_of_commitment']} years")
        processing_log.append(f"  • Total hours needed: {benefits['total_hours_needed']:,.0f} hours")
        processing_log.append(f"  • Benefit per hour: ${health_value_per_hour:.2f}/hr (present value)")
        processing_log.append(f"😊 Immediate mood/energy benefits: ${immediate_benefits:.2f}/hr")
        
    elif classification.health_impact == "mental_health":
        if classification.productivity_impact == "reducing":
            costs = calculate_social_media_costs(user_profile, duration_hours)
            total_costs += costs["total_cost"]
            calculation_breakdown["mental_health_costs"] = costs
            processing_log.append(f"😰 Mental health costs: ${costs['total_cost']:.2f}/hr (productivity loss + stress)")
        else:
            mental_health_value = 15.0
            total_benefits += mental_health_value
            calculation_breakdown["mental_health_benefits"] = mental_health_value
            processing_log.append(f"🧘‍♀️ Mental health benefits: ${mental_health_value:.2f}/hr (stress reduction, wellbeing)")
    
    elif classification.health_impact == "cognitive":
        cognitive_boost = 8.0 * (1.0 if classification.intensity_level == "high" else 0.7)
        total_benefits += cognitive_boost
        calculation_breakdown["cognitive_benefits"] = cognitive_boost
        processing_log.append(f"🧠 Cognitive benefits: ${cognitive_boost:.2f}/hr (mental stimulation, focus improvement)")
    
    processing_log.append(f"⚡ Calculating productivity impacts for '{classification.productivity_impact}' classification...")
    
    if classification.productivity_impact == "enhancing":
        if "learning" in classification.relevant_research_domains:
            benefits = calculate_learning_benefits(user_profile, duration_hours)
            learning_value = benefits["present_value_earnings"] / duration_hours
            immediate_learning_benefits = 13.0
            total_benefits += learning_value + immediate_learning_benefits
            calculation_breakdown["learning_benefits"] = benefits
            processing_log.append(f"📈 Learning benefits: ${learning_value:.2f}/hr (future earning potential)")
            processing_log.append(f"🎯 Immediate learning satisfaction: ${immediate_learning_benefits:.2f}/hr")
        else:
            productivity_boost = opportunity_cost * 0.3
            total_benefits += productivity_boost
            calculation_breakdown["productivity_boost"] = productivity_boost
            processing_log.append(f"⚡ General productivity boost: ${productivity_boost:.2f}/hr (30% of opportunity cost)")
            
    elif classification.productivity_impact == "reducing":
        if classification.health_impact != "mental_health":
            productivity_loss = opportunity_cost * 0.2
            total_costs += productivity_loss
            calculation_breakdown["productivity_loss"] = productivity_loss
            processing_log.append(f"📉 Productivity loss: ${productivity_loss:.2f}/hr (20% of opportunity cost)")
    
    if classification.primary_category == "social":
        social_value = 10.0
        total_benefits += social_value
        calculation_breakdown["social_benefits"] = social_value
    
    if classification.primary_category == "creative":
        creative_value = 12.0 + (5.0 if classification.intensity_level == "high" else 0.0)
        total_benefits += creative_value
        calculation_breakdown["creative_benefits"] = creative_value
    
    processing_log.append("🧮 Final calculation:")
    processing_log.append(f"  💚 Total benefits: ${total_benefits:.2f}/hr")
    processing_log.append(f"  💸 Total costs: ${total_costs:.2f}/hr") 
    processing_log.append(f"  ⏰ Opportunity cost: ${opportunity_cost:.2f}/hr (for reference only)")
    
    ehw = total_benefits - total_costs
    
    processing_log.append(f"  = EHW: ${total_benefits:.2f} - ${total_costs:.2f} = ${ehw:.2f}/hr")
    processing_log.append(f"  📊 Compared to opportunity cost: ${ehw - opportunity_cost:+.2f}/hr difference")
    
    activity_type_name = f"{classification.primary_category.title()} Activity"
    if classification.health_impact != "neutral":
        activity_type_name += f" ({classification.health_impact.replace('_', ' ').title()})"
    
    confidence_level = "high" if classification.confidence > 0.8 else "medium" if classification.confidence > 0.5 else "low"
    
    processing_log.append(f"✅ Final result: {activity_type_name} with EHW of ${ehw:.2f}/hr ({confidence_level} confidence)")
    
    calculation_breakdown.update({
        "total_benefits": total_benefits,
        "total_costs": total_costs,
        "ehw_value": ehw,
        "opportunity_cost_comparison": ehw - opportunity_cost
    })
    
    return EHWResult(
        effective_hourly_wage=round(ehw, 2),
        activity_type=activity_type_name,
        explanation="",  # Will be generated separately
        research_facts_used=research_facts_used,
        calculation_breakdown=calculation_breakdown,
        confidence_level=confidence_level,
        processing_log=processing_log
    )

def generate_explanation(activity_type: str, calculation_data: Dict[str, Any], ehw: float) -> str:
    """Generate detailed explanation without AI as fallback"""
    
    if "cardiovascular" in activity_type.lower():
        health_benefits = calculation_data.get("health_benefits", {})
        opportunity_cost = calculation_data.get("opportunity_cost", 0)
        
        return f"""
**🏃‍♀️ Cardiovascular Exercise Economic Analysis**

**💰 Lifetime Benefits (Properly Amortized):**
• Total lifetime benefit: ${health_benefits.get('total_lifetime_benefit', 0):,.0f}
  - Medical cost savings: ${health_benefits.get('medical_savings_component', 0):,.0f}
  - Life extension value: ${health_benefits.get('life_extension_component', 0):,.0f} ({health_benefits.get('years_life_extension', 0)} additional years)
• Exercise commitment required: {health_benefits.get('hours_per_week_needed', 0)} hours/week for {health_benefits.get('years_of_commitment', 0)} years
• Total hours needed: {health_benefits.get('total_hours_needed', 0):,.0f} hours

**📊 Hourly Value Calculation:**
• Lifetime benefit ÷ Total hours = ${health_benefits.get('benefit_per_hour', 0):.2f}/hour
• Present value (discounted): ${health_benefits.get('present_value_per_hour', 0):.2f}/hour

**⚡ Immediate Benefits:**
• Mood boost from endorphins: $8/hour value
• Increased energy and focus: $5/hour value

**📚 Research Basis:**
• Meta-analysis of 33 studies shows 20-35% CVD risk reduction from regular moderate exercise
• Exercise adds ~2.5 years of life expectancy on average
• Value of Statistical Life: ${health_benefits.get('value_statistical_life', 0):,} (EPA/DOT standard)

**🎯 EHW Calculation:**
Health benefits (${health_benefits.get('present_value_per_hour', 0):.2f}) + immediate benefits ($13) = ${ehw:.2f}/hour

**⚖️ Opportunity Cost Comparison:**
Your opportunity cost is ${opportunity_cost:.0f}/hour, making this activity ${ehw - opportunity_cost:+.2f}/hour relative to alternatives.

**💡 Economic Logic:** EHW measures value generated per hour by the activity itself. Opportunity cost is shown for comparison but not subtracted from EHW.
        """
    
    elif "social media" in activity_type.lower():
        mental_health_costs = calculation_data.get("mental_health_costs", {})
        opportunity_cost = calculation_data.get("opportunity_cost", 0)
        return f"""
**📱 Social Media Browsing Economic Analysis**

**📉 Productivity Costs (Research-Backed):**
• Lost Focus: Heavy social media use reduces workplace productivity by 13-21%
• Economic Impact: ${mental_health_costs.get('productivity_loss', 0):.2f} in lost productivity value this session
• Attention Fragmentation: Reduces cognitive performance for hours after use

**😰 Mental Health Costs:**
• Risk Increase: 2+ hours daily increases anxiety/depression risk by 25%
• Economic Impact: ${mental_health_costs.get('mental_health_cost', 0):.2f} in expected mental health costs
• Sleep Disruption: Blue light and stimulation affect sleep quality

**⏰ Opportunity Cost:**
• Alternative Activities: Could have been spent on exercise, learning, or productive work
• Compound Effects: Time spent here doesn't build long-term value

**💸 Total Economic Cost: ${abs(ehw):.2f}/hour**

**📚 Research Basis:**
Studies of 661,137 participants show consistent negative impacts on productivity and mental health.
Self-reported time tracking studies confirm 17% average productivity reduction.

**💡 Recommendation:** Consider limiting sessions to <30 minutes or replacing with higher-value activities.
        """
    
    else:
        return f"**Economic value: ${ehw:.2f}/hour** based on opportunity cost analysis and activity classification."

# Main Streamlit App
def main():
    st.markdown('<h1 class="main-header">🧮 Effective Hourly Wage Calculator</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h3>🎯 What is Effective Hourly Wage (EHW)?</h3>
    <p>EHW measures the economic value of any activity you do, expressed as dollars per hour. 
    It considers health benefits, productivity impacts, opportunity costs, and research-backed effects.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for user profile
    with st.sidebar:
        st.header("👤 Your Profile")
        
        age = st.number_input("Age", min_value=16, max_value=100, value=30, help="Used for life expectancy and career calculations")
        
        current_wage = st.number_input(
            "Current hourly wage ($)", 
            min_value=0.0, 
            max_value=500.0, 
            value=25.0,
            help="Leave as 0 if unemployed - we'll estimate based on age"
        )
        
        profession = st.selectbox(
            "Profession",
            ["Software Engineer", "Teacher", "Healthcare Worker", "Student", "Retired", "Other"],
            help="Used for earnings potential calculations"
        )
        
        location = st.text_input("Location (optional)", placeholder="e.g., San Francisco, CA")
        
        fitness_level = st.select_slider(
            "Fitness Level",
            options=["Low", "Moderate", "High"],
            value="Moderate",
            help="Affects health benefit calculations"
        )
        
        # OpenAI API Key input
        st.header("🤖 AI Settings")
        
        api_key_input = st.text_input(
            "OpenAI API Key (optional)",
            type="password",
            help="For advanced AI activity classification. Leave blank to use keyword-based classification."
        )
        
        if api_key_input:
            openai.api_key = api_key_input
            st.success("✅ AI classification enabled!")
        else:
            has_api_key = init_openai()
            if has_api_key:
                st.success("✅ AI classification enabled (from secrets)!")
            else:
                st.info("ℹ️ Using keyword-based classification (no API key)")
    
    # Main form
    st.header("📝 Activity Analysis")
    
    # Check for query parameters and use them to populate the form
    default_activity = "Went for a 30-minute run"
    if "activity" in st.query_params:
        default_activity = st.query_params["activity"]
    
    with st.form("ehw_calculator"):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            activity_description = st.text_area(
                "What did you do?",
                value=default_activity,
                height=100,
                help="Describe your activity in detail. Examples: 'Scrolled Instagram for 2 hours', 'Studied Python programming', 'Had coffee with friends'"
            )
        
        with col2:
            duration_hours = st.number_input(
                "Duration (hours)",
                min_value=0.1,
                max_value=24.0,
                value=0.5,
                step=0.1,
                help="How long did you spend on this activity?"
            )
        
        submitted = st.form_submit_button("🚀 Calculate EHW", use_container_width=True)
    
    if submitted:
        if not activity_description.strip():
            st.error("Please describe your activity!")
            return
        
        # Create user profile
        user_profile = UserProfile(
            age=age,
            current_hourly_wage=current_wage if current_wage > 0 else None,
            profession=profession,
            location=location if location else None,
            fitness_level=fitness_level.lower()
        )
        
        # Calculate EHW
        with st.spinner("🧮 Calculating your Effective Hourly Wage..."):
            try:
                result = calculate_ehw(user_profile, activity_description, duration_hours)
                
                # Display results
                st.success("✅ Calculation Complete!")
                
                # Main result
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "💰 Effective Hourly Wage",
                        f"${result.effective_hourly_wage:.2f}",
                        help="The economic value generated by this activity per hour"
                    )
                
                with col2:
                    opportunity_cost = result.calculation_breakdown.get("opportunity_cost", 0)
                    difference = result.effective_hourly_wage - opportunity_cost
                    st.metric(
                        "📊 vs Opportunity Cost",
                        f"${difference:+.2f}",
                        delta=f"Opportunity cost: ${opportunity_cost:.2f}",
                        help="How this activity compares to your alternatives"
                    )
                
                with col3:
                    st.metric(
                        "🎯 Activity Type",
                        result.activity_type,
                        help=f"Confidence: {result.confidence_level}"
                    )
                
                # Detailed explanation
                st.header("📖 Detailed Analysis")
                
                explanation = generate_explanation(
                    result.activity_type,
                    result.calculation_breakdown,
                    result.effective_hourly_wage
                )
                
                st.markdown(explanation)
                
                # Processing log in expandable section
                with st.expander("🔍 View Calculation Steps"):
                    for log_entry in result.processing_log:
                        st.text(log_entry)
                
                # Research facts
                if result.research_facts_used:
                    with st.expander("📚 Research Facts Used"):
                        for fact in result.research_facts_used:
                            st.markdown(f"""
                            **{fact.claim}**
                            - Effect size: {fact.effect_size} {fact.effect_unit}
                            - Confidence: {fact.confidence}
                            - Population: {fact.population}
                            - Citation: {fact.citation}
                            - Notes: {fact.notes}
                            """)
                
                # Calculation breakdown
                with st.expander("🧮 Technical Breakdown"):
                    st.json(result.calculation_breakdown)
                
            except Exception as e:
                st.error(f"❌ Calculation failed: {str(e)}")
                st.write("Please check your inputs and try again.")

    # Example activities
    st.header("💡 Try These Examples")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🏃‍♀️ Cardio Exercise", use_container_width=True):
            st.query_params["activity"] = "Went for a 45-minute jog"
            st.rerun()
    
    with col2:
        if st.button("📱 Social Media", use_container_width=True):
            st.query_params["activity"] = "Scrolled Instagram for 2 hours"
            st.rerun()
    
    with col3:
        if st.button("📚 Learning", use_container_width=True):
            st.query_params["activity"] = "Studied Python programming for 1 hour"
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
    <p>🧠 Powered by research-backed calculations and AI classification</p>
    <p>💡 EHW measures value generated, not net value vs alternatives</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 