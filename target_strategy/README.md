# Target Strategy Experiment

## Research question

Does the εeff << ε gap observed in prior work (Houssiau et al. 2022, Chida et al. 2024) persist across different target selection strategies, or is there a strategy that produces more reliable εeff estimates for auditing DP guarantees without underestimating privacy leakage?

## Motivation

Components defining the (d, d') neighbouring dataset pair include background dataset size, generator, and target record selection strategy. TAPAS used an outlier target (lowest log-likelihood from a 1000-record sample); Chida et al. used an artificial worst-case target (minimum-frequency value per attribute). Neither compared their choice against alternatives.

TAPAS reported εeff 95% CI = [0.86, 1.39] for MST at ε = 10. Three plausible explanations for the gap:
1. Loose proof (ε is not a tight upper bound)
2. Non-optimal (d, d') pair
3. Non-optimal attack

This experiment isolates factor 2 by varying only the target selection strategy.

## Experiment design

**Varied:** Target selection strategy 
- random
- outlier (Houssiau et al. 2022)
- artificial worst-case (Chida et al. 2024)

**Fixed:** 
- Adult Census dataset
- background dataset (499, randomly sampled once, seed=42)
- generator (Synthcity PrivBayes, ε=10)
- Attacker's ExactDataKnowledge, BlackBoxKnowledge
- all 5 TAPAS MIA attacks (Groundhog, ShadowModelling+RandomQueries, LocalNeighbourhood, ProbabilityEstimation, ClosestDistanceMIA)

**Reported:** 
- εeff per attack per strategy
- final εeff = max across attacks

**Scale:** 2.38 hours per strategy
- num_train=50
- num_test=100 

## Expected outcomes

- If a strategy produces εeff -> ε, researchers is recommended to use that strategy when auditing DP via εeff.
- If εeff stays near-zero regardless, target selection is ruled out as a contributing factor, narrowing the explanation to loose proofs and/or non-optimal attacks.

## Running target selection strategies

```bash
cd priv-sdg
conda activate priv-sdg
python target_strategy/strategies/random_strategy.py
python target_strategy/strategies/outlier_strategy.py
python target_strategy/strategies/artificial_strategy.py
```

## Folder structure
target_strategy/
├── README.md
├── config.py
├── common.py
├── analysis.ipynb
├── strategies/
│   ├── random_strategy.py
│   ├── outlier_strategy.py
│   └── artificial_strategy.py
├── cache/                        # gitignored
│   ├── random_strategy/
│   │   ├── threat_model.pkl
│   │   └── result_*.json
│   ├── outlier_strategy/
│   └── artificial_strategy/
└── results/                      
    ├── random_strategy/
    │   ├── effeps_random.csv
    │   ├── effective_epsilon_*.csv
    │   └── ROC_curve_*.png
    ├── outlier_strategy/
    └── artificial_strategy/