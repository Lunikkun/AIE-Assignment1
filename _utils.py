import json
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier


def _build_output_path(filename, subfolder):
    base_dir = os.path.join("results", subfolder)
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, filename)

def get_candidates(user_id, train_ratings, all_items):
    rated_items = set(train_ratings[train_ratings['user_id'] == user_id]['item_id'])
    candidates = [item for item in all_items if item not in rated_items]
    return candidates

def rank_popularity(candidates, train_ratings):
    popularity = train_ratings.groupby('item_id').size()
    scored_items = [(item, popularity.get(item, 0)) for item in candidates]
    scored_items.sort(key=lambda x: x[1], reverse=True)
    return [item for item, score in scored_items]

def rank_mf(candidates, user_id, mf_model, user_map, item_map, popularity=None):
    u_idx = user_map[user_id]
    scores = []
    for item in candidates:
        i_idx = item_map[item]
        if hasattr(mf_model, "predict") and mf_model.__class__.__name__.startswith("MF_"):
            score = float(mf_model.predict(user_map[user_id], item_map[item]))
        else:
            if popularity is None:
                raise ValueError("Popularity necessary for pairwise")
            user_emb = mf_model.user_factors[user_map[user_id]]
            item_emb = mf_model.item_factors[item_map[item]]
            pop = popularity[item_map[item]]
            X = np.concatenate([user_emb, item_emb, [pop]]).reshape(1, -1)
            score = float(mf_model.predict(X)[0])
        scores.append((item, score))
    scores.sort(key=lambda x: x[1], reverse=True)  
    return [item for item, score in scores]

def save_results(user_id, ranked_pop, ranked_mf, mf_model, movies, train_ratings, test_ratings, user_map, item_map, model_type, calculate_rmse=None, calculate_accuracy=None):
    results = {
        "user_id": int(user_id), 
        "popularity_top10": [
            {"item_id": int(item), "title": movies[movies['movie_id'] == item]['movie_title'].values[0] if movies[movies['movie_id'] == item].shape[0] > 0 else "N/A", "popularity": int(train_ratings[train_ratings['item_id'] == item].shape[0])}
            for item in ranked_pop[:10]
        ],
        "mf_top10": [
            {"item_id": int(item), "title": movies[movies['movie_id'] == item]['movie_title'].values[0] if movies[movies['movie_id'] == item].shape[0] > 0 else "N/A", "pred_score": float(mf_model.predict(user_map[user_id], item_map[item]))}
            for item in ranked_mf[:10]
        ],
        "metrics": {
            "final_train_loss": float(getattr(mf_model, 'best_loss', None)) if getattr(mf_model, 'best_loss', None) is not None else None
        },
        "model_params": {
            "n_factors": int(mf_model.n_factors),
            "reg": float(mf_model.reg),
            "lr": float(getattr(mf_model, 'lr', None)) if getattr(mf_model, 'lr', None) is not None else None, 
            "patience": int(mf_model.patience)
        }
    }
    
    results["model_type"] = model_type  
    
    filename = f'results_user{user_id}_{model_type}.json' 
    output_path = _build_output_path(filename, "legacy")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Saved in {output_path}")

def build_pairwise_data(train_ratings, mf_model, user_map, item_map, max_pairs_per_user=50):
    popularity = train_ratings.groupby('item_id').size().to_dict()
    rng = np.random.default_rng(42)
    X, y = [], []
    for u in train_ratings['user_id'].unique():
        user_rows = train_ratings[train_ratings['user_id'] == u]
        rows_list = list(user_rows.itertuples(index=False))
        u_idx = user_map[u]

        pairs = [
            (i_row, j_row)
            for idx_i, i_row in enumerate(rows_list)
            for j_row in rows_list[idx_i + 1:]
            if i_row.rating != j_row.rating
        ]

        if len(pairs) > max_pairs_per_user:
            chosen = rng.choice(len(pairs), size=max_pairs_per_user, replace=False)
            pairs = [pairs[k] for k in chosen]

        for i_row, j_row in pairs:
            i_idx = item_map[i_row.item_id]
            j_idx = item_map[j_row.item_id]
            fi = np.array([
                mf_model.predict(u_idx, i_idx),
                mf_model.bu[u_idx],
                mf_model.bi[i_idx],
                popularity.get(i_row.item_id, 0),
                1.0
            ])
            fj = np.array([
                mf_model.predict(u_idx, j_idx),
                mf_model.bu[u_idx],
                mf_model.bi[j_idx],
                popularity.get(j_row.item_id, 0),
                1.0
            ])
            if i_row.rating > j_row.rating:
                X.append(fi - fj); y.append(1)
                X.append(fj - fi); y.append(0)
            else:
                X.append(fj - fi); y.append(1)
                X.append(fi - fj); y.append(0)
    return np.vstack(X), np.array(y), popularity

def train_pairwise_ranker(train_ratings, mf_model, user_map, item_map, max_epochs=8):
    X, y, popularity = build_pairwise_data(train_ratings, mf_model, user_map, item_map)
    model = SGDClassifier(
        loss="log_loss",
        max_iter=max_epochs,
        tol=None,
        learning_rate="constant",
        eta0=0.01,
        random_state=42,
    )
    model.fit(X, y)
    return model, popularity


def save_pairwise_artifacts(path, pairwise_model, popularity):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"pairwise_model": pairwise_model, "popularity": popularity}, f)


def load_pairwise_artifacts(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["pairwise_model"], data["popularity"]

def rank_pairwise(candidates, user_id, mf_model, pairwise_model, user_map, item_map, popularity):
    u_idx = user_map[user_id]
    scored=[]
    for item in candidates:
        i_idx=item_map[item]
        f=np.array([
            mf_model.predict(u_idx,i_idx),
            mf_model.bu[u_idx],
            mf_model.bi[i_idx],
            popularity.get(item,0),
            1.0
        ])
        p=pairwise_model.predict_proba([f])[0,1]
        scored.append((item, p))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [item for item,_ in scored]

def save_results_mf(user_id, ranked_pop, ranked_mf, mf_model, movies, train_ratings, test_ratings, user_map, item_map, model_type):
    results = {
        "user_id": int(user_id),
        "popularity_top10": [
            {"item_id": int(item), "title": movies[movies['movie_id'] == item]['movie_title'].values[0] if movies[movies['movie_id'] == item].shape[0] > 0 else "N/A", "popularity": int(train_ratings[train_ratings['item_id'] == item].shape[0])}
            for item in ranked_pop[:10]
        ],
        "mf_top10": [
            {"item_id": int(item), "title": movies[movies['movie_id'] == item]['movie_title'].values[0] if movies[movies['movie_id'] == item].shape[0] > 0 else "N/A", "pred_score": float(mf_model.predict(user_map[user_id], item_map[item]))}
            for item in ranked_mf[:10]
        ],
        "metrics": {
            "final_train_loss": float(getattr(mf_model, 'best_loss', None)) if getattr(mf_model, 'best_loss', None) is not None else None
        },
        "model_params": {
            "n_factors": int(mf_model.n_factors),
            "reg": float(mf_model.reg),
            "lr": float(getattr(mf_model, 'lr', None)) if getattr(mf_model, 'lr', None) is not None else None,
            "patience": int(mf_model.patience)
        },
        "model_type": model_type
    }
    filename = f'results_user{user_id}_{model_type}.json'
    output_path = _build_output_path(filename, "mf")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved in {output_path}")


def save_results_mmr(user_id, mmr_results_by_alpha, base_ranker_name, movies, train_ratings):
    results = {
        "user_id": int(user_id),
        "base_ranker": base_ranker_name,
        "mmr_top10_by_alpha": {}
    }

    for alpha, ranked in mmr_results_by_alpha.items():
        results["mmr_top10_by_alpha"][f"alpha_{alpha}"] = [
            {
                "item_id": int(item),
                "title": movies[movies['movie_id'] == item]['movie_title'].values[0] if movies[movies['movie_id'] == item].shape[0] > 0 else "N/A",
                "popularity": int(train_ratings[train_ratings['item_id'] == item].shape[0])
            }
            for item in ranked[:10]
        ]

    filename = f'results_user{user_id}_mmr_{base_ranker_name}.json'
    output_path = _build_output_path(filename, f"mmr/{base_ranker_name}")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved in {output_path}")


def save_results_pairwise(user_id, ranked_pop, ranked_pairwise, pairwise_model, base_mf_model, popularity, movies, train_ratings, user_map, item_map, model_type):
    def get_pairwise_score(item):
        u_idx = user_map[user_id]
        i_idx = item_map[item]
        f = np.array([
            base_mf_model.predict(u_idx, i_idx),
            base_mf_model.bu[u_idx],
            base_mf_model.bi[i_idx],
            popularity.get(item, 0),
            1.0
        ])
        return float(pairwise_model.predict_proba([f])[0, 1])

    results = {
        "user_id": int(user_id),
        "popularity_top10": [
            {"item_id": int(item), "title": movies[movies['movie_id'] == item]['movie_title'].values[0] if movies[movies['movie_id'] == item].shape[0] > 0 else "N/A", "popularity": int(train_ratings[train_ratings['item_id'] == item].shape[0])}
            for item in ranked_pop[:10]
        ],
        "mf_top10": [
            {"item_id": int(item), "title": movies[movies['movie_id'] == item]['movie_title'].values[0] if movies[movies['movie_id'] == item].shape[0] > 0 else "N/A", "pred_score": get_pairwise_score(item)}
            for item in ranked_pairwise[:10]
        ],
        "model_type": model_type
    }
    filename = f'results_user{user_id}_{model_type}.json'
    output_path = _build_output_path(filename, "pairwise")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved in {output_path}")
