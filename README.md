# MovieLens 100K Recommender System - Results

### Models
- **MF-SGD**: Stochastic Gradient Descent (10 factors, L2 regularization)
- **MF-ALS**: Alternating Least Squares (10 factors)
- **Pairwise LTR**: LogisticRegression on pairwise feature differences

### Rankers
1. **R1 - Popularity**: Items ranked by frequency in training set
2. **R2 - MF-SGD**: Predicted ratings via MF-SGD model
3. **R3 - MF-ALS**: Predicted ratings via MF-ALS model
4. **R4 - Pairwise LTR**: Item-pair preferences via trained ranker (2 variants: on SGD or ALS base)
5. **R5 - MMR**: Maximal Marginal Relevance reranking with tunable α ∈ {0.1, 0.4, 0.7}
   - Relevance: position in base ranking
   - Diversity: cosine distance on L2-normalized genre vectors
   - Formula: c* = argmax_{c ∈ C\S} [(1-α)rel(c) - α max_s sim(c,s)]

### Personalization
- **Method**: EMA state vector updates per round
- **Formula**: u_{t+1} = normalize((1-ρ)u_t + ρv_i) where ρ ∈ {0.2, 0.4, 0.5}
- **State vector**: MF latent factors (dimension 10) or genre vectors
- **Sessions**: 3 independent users × 3 sessions × 5 rounds each 

## How to Reproduce

### Prerequisite: Venv and libs

python -m venv ./venv | pip install -r requirements.txt

### Step 1: Train models & generate rankings

python cocreation_loop1.py

### Step 2: Evaluate metrics

python report_metrics.py

### Step 3: Create table

python generate_results_tables.py



