Effective Hourly Wage (EHW) Calculator — System Architecture & Implementation Roadmap

Author: Principal Product Archetype
Audience: AI Engineering & Applied Research Team
Version: v0.1 (Architecture Draft)

⸻

1. Problem Statement & Product Mission

People routinely engage in activities whose true economic & well‑being impacts are opaque. The EHW Calculator converts any self‑described activity (“doomscrolling Twitter”, “45‑min Zone 2 run”, “day trading my personal account”) plus a minimal user profile into:
	1.	Point Estimate: Effective Hourly Wage (value/hour; can be negative).
	2.	Confidence Band: Distributional view (e.g. P5 / Median / P95).
	3.	Transparent Explanation: Step‑by‑step back‑of‑the‑envelope (BOTE) derivation referencing empirical research & parameter assumptions.
	4.	Externality Awareness: Decomposition of private (to user) vs societal externalities (positive / negative) when inferable.

Goal: Foster mindful time allocation by reframing activities in unified value units ($/hr or utility points/hr) with explicit uncertainty.

⸻

2. Core Concepts & Definitions

Term	Definition
EHW (Effective Hourly Wage)	Net_Value / Time_Hours where Net_Value = (Immediate_Benefits + Future_Benefits + Option_Value + External_Positive) - (Direct_Costs + Opportunity_Cost + External_Negative + Risk_Adjustment) measured in chosen Value Unit (default = USD equivalence).
Value Unit	Default: USD via shadow pricing of non-monetary outcomes (e.g., health QALY gains * QALY_Value). Alternate: abstract “util” scale.
User Baseline	Distribution over plausible alternative uses of the same time (activity taxonomy). Drives opportunity cost.
Activity Archetype	Parameter template (e.g., Aerobic Exercise, Passive Scrolling, Focused Deep Work, Speculative Trading) with priors over benefit/cost parameters.
Parameter Priors	Bayesian priors (e.g., improvement in cardiovascular risk per hour of moderate exercise) sourced from structured research metadata.
Research Fact	Normalized claim extracted from literature with effect size, variance, population, context, citation ID.
BOTE Step	Deterministic or sampled computation node in explanation graph.


⸻

3. High-Level Architecture Overview

┌──────────────────────────────────────────────────────────────────┐
|                   Client (Web / Mobile / API)                    |
|  - Q1: "Tell me about yourself" UI                              |
|  - Q2: "Describe the activity" free text & tagging               |
└──────────────┬──────────────────────────────────────────────────┘
               │ GraphQL / REST
┌──────────────▼──────────────────────────────────────────────────┐
|                  Orchestration Service                          |
|  - Request Validator                                            |
|  - Session & Trace ID                                           |
|  - Pipeline DAG Builder                                         |
|  - LLM Tool Router (Structured vs Generative)                   |
└───────┬───────────┬───────────────┬───────────────┬────────────┘
        │           │               │               │
        │           │               │               │
┌───────▼┐   ┌──────▼─────┐  ┌──────▼─────┐  ┌──────▼──────┐  ┌─────────▼─────────┐
| Profile|   | NLP &      |  | Research   |  | Valuation   |  | Simulation Engine  |
| Layer  |   | Activity   |  | Retrieval  |  | Engine      |  | (MC/Bayesian)      |
| (User  |   | Parsing    |  | & Fact     |  | (Determin.) |  | - Posterior EHW    |
| Priors)|   | & Typing   |  | Normalizer |  | + BOTE DAG  |  | - Sensitivity      |
└───┬────┘   └────┬───────┘  └────┬───────┘  └────┬────────┘  └─────────┬─────────┘
    │             │               │              │                        │
    │             │               │              │                        │
    ▼             ▼               ▼              ▼                        ▼
┌────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐             ┌───────────┐
| Storage|   | Embeddings|   | Fact KB  |   | BOTE DAG |             | Explain.  |
| (User  |   | / Vector  |   | (Effect  |   | Graph    |             | Generator |
| Config)|   | Index     |   | Sizes)   |   | Nodes    |             | (LLM)     |
└────────┘   └──────────┘   └──────────┘   └──────────┘             └───────────┘


⸻

4. Modular Components

4.1 Profile Layer
	•	Inputs: Free-text self description (age, profession, income bracket, health goals, risk tolerance, baseline hourly wage if any, constraints).
	•	Processing:
	•	Structured extraction via LLM -> JSON schema (Pydantic): Demographics, Income.current, Income.target, HealthMetrics, TimeBudget, BaselineActivitiesDistribution.
	•	Estimation of Opportunity Cost Distribution (OCD): mixture over baseline activities with expected value/hr.
	•	Persist hashed user ID; PII minimized (privacy requirements §10).

4.2 NLP & Activity Parsing
	•	Goal: Map raw activity text to: Archetype, Intensity, Duration, Context (alone / social / professional), Intent (intrinsic / extrinsic), Outcome_Horizon.
	•	Pipeline:
	1.	Text normalization & profanity / safety filtering.
	2.	LLM classification against controlled taxonomy (stored as YAML; versioned).
	3.	Parameter guess (default priors) + user confirmation loop (optional) for missing numeric fields (e.g. duration if absent).

4.3 Research Retrieval & Fact Normalizer
	•	Sources: Curated internal fact base + public literature (PubMed / meta-analyses). Architecture provides interface; ingestion pipeline separate.
	•	Data Model (Fact): {fact_id, archetype, effect_metric, effect_size_distribution (mean, std / distribution type), population_tags[], dose_response_fn(optional), citation_meta, notes}.
	•	Retrieval: Semantic search (vector index) + symbolic filters (archetype match, population relevance). Score & return top-K.
	•	Normalization: Convert heterogeneous effect metrics into standard value inputs, e.g. ΔQALY/hour, ΔProductivity%/hour, ΔStressScore, ExpectedReturn%.
	•	Shadow Pricing: Map normalized metrics to USD using conversion modules (e.g. QALY_to_USD, StressPoint_to_USD). Each module contains: assumptions, parameter priors, references.
	•	Transparency: All applied facts and conversions enumerated in DAG.

4.4 Valuation Engine & BOTE DAG Builder
	•	Objective: Construct a composable computation graph where each node is typed:
	•	InputNode (User / Fact / Prior)
	•	TransformNode (e.g. dose-response scaling, time discounting)
	•	AggregationNode (sum, expected value)
	•	AdjustmentNode (risk, externality)
	•	OutputNode (EHW).
	•	Representation: Directed acyclic graph (JSON serializable). Example snippet:

{
  "nodes": [
    {"id": "n1", "type": "Input", "name": "BaseOpportunityCost", "distribution": {"lognormal": {"mu": 3.4, "sigma": 0.5}}, "unit": "$ / hr"},
    {"id": "n2", "type": "Input", "name": "CardioRiskReduction", "distribution": {"normal": {"mean": 0.0008, "sd": 0.0002}}, "unit": "QALY / hr"},
    {"id": "n3", "type": "Transform", "fn": "QALY_to_USD", "input": "n2", "params": {"usd_per_qaly": {"dist": {"normal": {"mean": 150000, "sd": 20000}}}}},
    {"id": "n4", "type": "Aggregation", "inputs": ["n3"], "fn": "sum"},
    {"id": "n5", "type": "Adjustment", "fn": "subtract", "inputs": ["n4", "n1"]},
    {"id": "n6", "type": "Output", "formula": "n5 / ActivityDurationHours"}
  ],
  "globals": {"ActivityDurationHours": 1.0}
}

	•	Extensibility: Adding new conversion (e.g. Cognitive_Improvement_to_USD) = new TransformNode function with metadata & tests.

4.5 Simulation Engine (Uncertainty & Confidence Band)
	•	For each stochastic node, sample from defined distributions (Monte Carlo, N=5k default) to produce EHW distribution.
	•	Optional: Use analytical propagation for linear combinations to speed up; fallback to sampling for nonlinear transforms.
	•	Offers Sensitivity Analysis (Sobol indices or variance-based) to rank parameter influence => feed into explanation & UI.
	•	Supports Bayesian updating if the user supplies observed outcomes (future iteration).

4.6 Explanation Generator
	•	Consumes DAG + sample statistics.
	•	Constructs Reasoning Trace: Ordered list of BOTE steps (template-driven).
	•	LLM fills narrative templates with structured slots (numbers, ranges, cited facts) to reduce hallucination.
	•	Each claim anchored to node IDs & citation_meta.
	•	Exposes dual views:
	1.	Narrative: Human friendly prose.
	2.	Trace Table: Tabular nodes (name, mean, unit, P5, P95, source).
	•	Includes automated plausibility guardrails (e.g., flag if EHW > 10× baseline wage or < -baseline*5 unless justified by tail risk nodes).

4.7 Externality Module
	•	Distinguish Private Value vs External Value per node tagging.
	•	Present stacked bar: contributions to EHW (private vs external) & net sign.
	•	Provide disclaimers where evidence is sparse (low evidence score).

4.8 Persistence & Observability
	•	Store anonymized request + DAG + distribution summary for offline calibration, with user consent.
	•	Metrics: latency_ms, token_usage, explanation_tokens, evidence_density (citations / 100 words), hallucination_flags.
	•	Feature store for learned personal adjustments (future stage).

4.9 API Layer
	•	Endpoints (REST/GraphQL):
	•	POST /evaluate (payload: user_profile_text, activity_text, options{currency, confidence_levels}) → returns EHWEstimate object.
	•	GET /explanation/{trace_id}.
	•	GET /taxonomy.
	•	POST /feedback (user rating plausibility, correctness signals).
	•	JSON Schemas: Versioned. Breaking changes flagged.

⸻

5. Data & Distribution Modeling

5.1 Distribution Types

Type	Usage	JSON Spec Example
Normal	Small effect size uncertainties	{ "normal": {"mean": 0.05, "sd": 0.01}}
LogNormal	Income / opportunity cost	{ "lognormal": {"mu": 3.2, "sigma": 0.4}}
Beta	Probabilities (success rate)	{ "beta": {"alpha": 2, "beta": 5}}
Triangular	Expert elicitation quick prior	{ "triangular": {"low": -10, "mode": 5, "high": 15}}
PointMass	Deterministic	{ "point": {"value": 0.0}}

5.2 Dose-Response & Time Scaling
	•	Functions: linear, diminishing returns (Michaelis-Menten), threshold, U-shaped risk (e.g., sleep).
	•	Encode function type & params in TransformNode; permit composition (exercise -> health risk reduction -> QALY -> USD).

5.3 Currency & Purchasing Power
	•	FX & inflation service (Stage 3) to localize values. Default USD for internal modeling.

⸻

6. Confidence Band Computation
	1.	Build DAG with distributions.
	2.	Deterministic topological order generation.
	3.	Monte Carlo sampling with vectorized ops (NumPy / JAX).
	4.	Compute summary statistics: mean, median, P5, P25, P75, P95.
	5.	Fit optional kernel density for smooth UI chart.
	6.	Compute Value at Risk style tail metrics (e.g., P10 negative loss) for risk communication.
	7.	Sensitivity: Partial derivatives (if differentiable) or variance decomposition to produce ranking list.

⸻

7. Explanation Strategy (Anti‑Hallucination)

Step	Mechanism	Notes
Structured Extraction	LLM → strict JSON; invalid JSON triggers repair loop (syntactic + semantic validation).	Pydantic for enforcement.
Citation Binding	Every research fact holds citation_id; explanation builder refuses to surface a claim lacking a bound ID.	Enforce evidence_coverage >= threshold.
Numeric Grounding	All numbers inserted from DAG values only; no free numeric generation by LLM.	Prevents fabricated stats.
Uncertainty Language	Templated hedging based on coefficient of variation & evidence score.	Maps intervals to adjectives.
Counterfactual Check	Provide user with baseline alternative summary (“Had you done X instead…”).	Increases reflection.


⸻

8. Testing & Validation

8.1 Unit Tests
	•	Function-level: distributions, transform functions, currency conversion, dose-response evaluation.
	•	DAG validation: cycle detection, dimension/unit consistency.

8.2 Property-Based Tests
	•	Random DAG generation verifying invariants (no negative variance, idempotent serialization).

8.3 Golden Trace Tests
	•	Predefined scenario fixtures (e.g., 30 min vigorous run, 2 hrs doomscrolling, 1 hr skill learning).
	•	Store expected EHW interval ranges (broad at first; tighten over time as priors calibrate).

8.4 Simulation Consistency
	•	Analytical vs Monte Carlo means for linear subgraphs within tolerance.

8.5 Explanation Integrity
	•	Assert each numeric in narrative appears in trace table with matching value ± rounding tolerance.

8.6 Hallucination Guards
	•	Red-team prompts for forcing LLM to invent sources; expect refusal & error flags.

8.7 Human Review Loop
	•	Sampling of sessions flagged by anomaly detector (EHW magnitude z-score > 3) for manual audit.

8.8 Calibration
	•	Over time, collect user-reported realized outcomes (optional) to perform posterior predictive checks & recalibrate priors.

⸻

9. Implementation Stages & Roadmap

Stage	Duration (wks)	Goal	Scope	Exit Criteria
0. Foundations	1	Repo, CI/CD, schemas	Monorepo setup, Pydantic models, taxonomy YAML skeleton	CI green; schema version v0.1 published
1. MVP Deterministic	2	Basic EHW calc without uncertainty	Profile extraction, activity typing, static parameter tables, single explanation template	3 golden scenarios produce plausible EHW & explanation
2. Uncertainty Engine	2	Monte Carlo + intervals	Distribution spec, sampling engine, interval rendering, sensitivity basics	Confidence bands for golden scenarios stable across runs
3. Research Ingestion	3	Structured fact KB	ETL pipeline (manual curated subset), effect normalization, citation binding	≥50 facts across 5 archetypes with variance captured
4. Node-Based BOTE DAG	2	Replace ad-hoc calc with formal graph	DAG JSON, topological evaluation, explanation reading from DAG	DAG visible in debug UI & serialized
5. Advanced Explanations	2	Narrative generation improvements	Template slots, uncertainty language, counterfactual baseline	Qualitative review: clarity score ≥ threshold
6. Externality Module	2	Private vs societal value decomposition	Tagging system, stacked bars UI, external cost priors (e.g., carbon for driving)	Externalities visible for 3 archetypes
7. Calibration & Feedback	3	User feedback loop & anomaly detection	Feedback endpoint, z-score alerts, parameter updates (manual)	Weekly calibration report auto-generated
8. Personalization	4	Bayesian updating per user	Posterior updates, personal baselines, opt-in data storage UI	Reduction in prediction error vs prior (A/B)
9. Expansion	ongoing	More archetypes & conversions	Sleep, meditation, creative ideation, volunteering, social time, etc.	Coverage: 80% of top requested activities


⸻

10. Privacy, Ethics, Safety
	•	Minimize PII: Hash user id; discard raw free text after structured extraction unless consent to retain.
	•	Explain Uncertainty: Never present single number without interval; label low-evidence priors.
	•	Non-Financial Advice Disclaimer: Speculative trading valuations flagged with high risk disclaimers.
	•	Bias Mitigation: Monitor differential parameter usage across demographics; avoid embedding biased wage assumptions; allow user-provided baseline wage override.
	•	Data Governance: Version control for fact KB; signed commits for provenance.

⸻

11. Technology Stack

Layer	Tech Choices	Rationale
Backend Core	Python (FastAPI)	Async, Pydantic integration
Orchestration / DAG	Custom + networkx or lightweight internal	Fine-grained control & inspectability
Simulation	NumPy / JAX (future)	Vectorization & potential GPU
Storage	Postgres (structured), S3/Blob (DAG logs)	Reliability & audit trail
Vector Index	Qdrant / Weaviate / Elastic vector	Hybrid semantic + filter search
LLM Access	OpenAI / Anthropic / Local fallback	Quality vs cost trade-offs
Frontend	React + TypeScript	Standard, component ecosystem
Auth	JWT (user, session)	Simplicity for MVP
CI/CD	GitHub Actions + pytest + mypy	Quality gate
Infra	Docker + Terraform	Reproducible deployments


⸻

12. Data Schemas (Sketch)

# pydantic style (abridged)
class UserProfile(BaseModel):
    user_id: str
    demographics: Demographics
    baseline_hourly_wage: DistributionSpec | None
    opportunity_cost: DistributionSpec | None  # derived mixture

class ActivityRequest(BaseModel):
    raw_text: str
    parsed: ActivityParsed | None

class Fact(BaseModel):
    fact_id: str
    archetype: str
    effect_metric: str  # e.g. 'QALY_per_hour'
    effect_distribution: DistributionSpec
    population: list[str]
    citation: CitationMeta
    dose_response: DoseResponse | None
    evidence_score: float  # 0-1

class DAGNode(BaseModel):
    id: str
    type: Literal['Input','Transform','Aggregation','Adjustment','Output']
    formula: str | None
    inputs: list[str] | None
    distribution: DistributionSpec | None
    unit: str | None
    metadata: dict[str, Any]


⸻

13. Calculation Recipe Examples

13.1 Positive Archetype: 1 hr Moderate Cardio

OpportunityCost ~ LogNormal(μ,σ)  (baseline mixture)
CardioRiskReduction_QALY ~ Normal(mean=0.0008, sd=0.0002)
ValuePerQALY ~ Normal(150000, 20000)
FutureHealthValue = CardioRiskReduction_QALY * ValuePerQALY / (1 + discount_rate)^{years_to_realization}
ImmediateMoodBoost_Value ~ Triangular(5, 12, 25)  # $ equivalent
NetValue = FutureHealthValue + ImmediateMoodBoost_Value - OpportunityCost
EHW = NetValue / 1 hour

13.2 Negative Archetype: 2 hrs Doomscrolling

OpportunityCost (deep work alt) ~ LogNormal
LostProductivity_Value ~ Normal(80, 20)
StressIncrease_to_QALY_Loss ~ Normal(0.0001, 0.00005)
QALY_Value ~ Normal(150000,20000)
FutureHealthCost = StressIncrease_to_QALY_Loss * QALY_Value
NetValue = - LostProductivity_Value - FutureHealthCost - (OpportunityCost * 2)
EHW = NetValue / 2

13.3 Ambiguous: 1 hr Personal Trading

OpportunityCost ~ LogNormal
ExpectedReturn_Alpha ~ Normal(0.0, 0.05)  # per hour capital efficiency proxy
CapitalAtRisk = user.stated_capital * leverage_factor
RiskPenalty = VaR_95(CapitalAtRisk * ReturnDistribution)
SkillBuilding_Value ~ Triangular(-5, 10, 40)
Entertainment_Value ~ Triangular(0, 5, 15)
NetValue = ExpectedReturn_Alpha*CapitalAtRisk + SkillBuilding_Value + Entertainment_Value - OpportunityCost - RiskPenalty
EHW = NetValue / 1 hr


⸻

14. Sensitivity & Debugging Tools
	•	/debug/dag/{trace_id} returns: graph JSON + parameter table.
	•	/debug/sensitivity/{trace_id} returns: sorted variance contribution list.
	•	Command Line Tool: ehw simulate --scenario cardio_run.json --n 10000.
	•	Notebook Templates: For analysts to tune priors / shadow pricing modules.

⸻

15. Extension Hooks

Hook	Purpose	Example
register_transform(fn_meta)	Add new conversion	CognitiveScore → Productivity%
register_archetype(template)	New activity archetype	Meditation
register_explanation_template(id, template_str)	A/B test messaging	“Scientific Deep Dive” vs “Quick Summary”
feedback_consumer	Online learning	Adjust prior means


⸻

16. Open Questions & Risks

Area	Risk	Mitigation
Shadow Pricing Validity	Arbitrary USD conversions	Publish assumptions; allow toggling to non-monetary units
Research Quality	Low-quality single studies	Weight by evidence score; prefer meta-analyses
LLM Hallucination	Fabricated citations	Hard enforcement: only allow citations from Fact KB IDs
User Misinterpretation	Treating EHW as guaranteed earnings	Prominent uncertainty framing & disclaimers
Privacy	Storing sensitive health info	Optional fields; encryption at rest; data minimization


⸻

17. Definition of Done (Initial Public Beta)
	•	Supports ≥10 archetypes covering >70% of user submissions.
	•	Every output includes point estimate + P5/P50/P95.
	•	Explanation lists ≥3 cited facts where applicable or states “Insufficient direct research — using heuristic priors”.
	•	Latency p95 < 4s for typical request.
	•	<1% sessions flagged with unresolved hallucination errors.

⸻

18. Developer Onboarding Checklist
	1.	Clone repo & run make bootstrap (installs env, pre-commit, mypy).
	2.	make test → all tests pass.
	3.	Run docker compose up → local API at localhost:8000.
	4.	Submit archetype PR referencing design doc template.
	5.	Add at least one golden scenario test if introducing new valuation module.

⸻

19. Future Research Integrations (Roadmap Beyond v1)
	•	Personal wearable data ingestion (heart rate, sleep) to refine priors.
	•	Multi-activity portfolio optimization (time allocation recommendations).
	•	Social network externality modeling (group exercise positive spillovers).
	•	Carbon footprint cost integration for travel activities.

⸻

20. Appendix: Configuration Example

valuation:
  usd_per_qaly: { distribution: { normal: { mean: 150000, sd: 20000 } } }
  stress_point_usd: { point: { value: 3.5 } }
  discount_rate_annual: { point: { value: 0.03 } }
explanation:
  style: "concise"
  confidence_levels: [0.05, 0.5, 0.95]
retrieval:
  top_k: 5
  min_evidence_score: 0.4
simulation:
  n_samples: 5000
  random_seed: 42


⸻

End of Architecture Document