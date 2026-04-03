import os
import numpy as np
import pandas as pd

from MF_SGD import MF_SGD
from MF_ALS import MF_ALS
from _utils import (
    get_candidates,
    rank_popularity,
    rank_mf,
    save_results_mf,
    save_results_pairwise,
    save_results_mmr,
    train_pairwise_ranker,
    rank_pairwise,
    save_pairwise_artifacts,
)
from mmr import build_genre_matrix, mmr_rerank_from_ranking
from metrics import build_relevant_items_map
from personalization_ema import run_personalization_demo

np.random.seed(42)

FAST_MODE        = False
N_FACTORS        = 20
REG_LAMBDA       = 0.05
SGD_LR           = 0.01
SGD_EPOCHS       = 10
ALS_ITERS        = 8
PAIRWISE_EPOCHS  = 8
MMR_TOP_M        = 80
FAST_USER_SUBSET = 300

def load_data():
    print("[LOAD] Reading MovieLens 100K dataset...")
    
    ratings = pd.read_csv(
        './ml-100k/u.data', 
        sep='\t', 
        names=['user_id', 'item_id', 'rating', 'timestamp']
    )
    
    movies = pd.read_csv(
        './ml-100k/u.item',
        sep='|',
        encoding='latin-1',
        usecols=range(24),
        names=['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL']
        + [f'genre_{i}' for i in range(19)]
    )
    
    train_ratings = pd.read_csv(
        './ml-100k/u1.base',
        sep='\t',
        names=['user_id', 'item_id', 'rating', 'timestamp']
    )
    test_ratings = pd.read_csv(
        './ml-100k/u1.test',
        sep='\t',
        names=['user_id', 'item_id', 'rating', 'timestamp']
    )
    
    user_ids = ratings['user_id'].unique()
    item_ids = ratings['item_id'].unique()
    user_map = {uid: i for i, uid in enumerate(user_ids)}
    item_map = {iid: i for i, iid in enumerate(item_ids)}
    all_items = sorted(train_ratings['item_id'].unique())
    
    print(f"[LOAD] Users: {len(user_ids)}, Items: {len(item_ids)}, Train rows: {len(train_ratings)}, Test rows: {len(test_ratings)}")
    
    return ratings, movies, train_ratings, test_ratings, user_ids, item_ids, user_map, item_map, all_items

def stage_baseline(candidates_user, train_ratings, movies):
    ranked = rank_popularity(candidates_user, train_ratings)
    print(f"[R1] Top 10:")
    for i, item in enumerate(ranked[:10], 1):
        pop = train_ratings[train_ratings['item_id'] == item].shape[0]
        movie_title = movies[movies['movie_id'] == item]['movie_title'].values
        title = movie_title[0] if len(movie_title) > 0 else "N/A"
    return ranked

def stage_matrix_factorization(num_users, num_items, train_ratings, user_map, item_map):
    n_factors = N_FACTORS
    sgd_epochs = SGD_EPOCHS
    als_epochs = ALS_ITERS

    print(f"[R2] MF-SGD (n_factors={n_factors}, epochs={sgd_epochs})...")
    mf_sgd_model = MF_SGD(
        num_users,
        num_items,
        n_factors=n_factors,
        reg=REG_LAMBDA,
        lr=SGD_LR,
        patience=sgd_epochs + 1,
    )
    mf_sgd_model.train(train_ratings, user_map, item_map, epochs=sgd_epochs)
    mf_sgd_model.save("results/models/mf_sgd_u1.npz")

    print(f"[R3] MF-ALS (n_factors={n_factors}, epochs={als_epochs})...")
    mf_als_model = MF_ALS(
        num_users,
        num_items,
        n_factors=n_factors,
        reg=REG_LAMBDA,
        patience=als_epochs + 1,
    )
    mf_als_model.train(train_ratings, user_map, item_map, epochs=als_epochs)
    mf_als_model.save("results/models/mf_als_u1.npz")

    return mf_sgd_model, mf_als_model

def stage_pairwise_ltr(train_ratings, mf_sgd_model, mf_als_model, user_map, item_map):

    print("[R4.SGD] Pairwise LTR on MF-SGD...")
    pairwise_model_sgd, popularity_sgd = train_pairwise_ranker(
        train_ratings, mf_sgd_model, user_map, item_map, max_epochs=PAIRWISE_EPOCHS
    )
    save_pairwise_artifacts("results/models/pairwise_sgd_u1.pkl", pairwise_model_sgd, popularity_sgd)
    print(f"[R4.SGD] Saved")
    
    print("[R4.ALS] Pairwise LTR on MF-ALS...")
    pairwise_model_als, popularity_als = train_pairwise_ranker(
        train_ratings, mf_als_model, user_map, item_map, max_epochs=PAIRWISE_EPOCHS
    )
    save_pairwise_artifacts("results/models/pairwise_als_u1.pkl", pairwise_model_als, popularity_als)
    print(f"[R4.ALS] Saved")
    
    return pairwise_model_sgd, popularity_sgd, pairwise_model_als, popularity_als

def stage_save_base_rankings(
    user_id, ranked_pop, ranked_sgd, ranked_als, ranked_pw_sgd, ranked_pw_als,
    mf_sgd_model, mf_als_model, popularity_sgd, popularity_als,
    pairwise_model_sgd, pairwise_model_als,
    movies, train_ratings, test_ratings, user_map, item_map
):
    print("\n[SAVE] Saving base ranker results...")
    save_results_mf(user_id, ranked_pop, ranked_sgd, mf_sgd_model, movies, train_ratings, test_ratings, user_map, item_map, "sgd")
    save_results_mf(user_id, ranked_pop, ranked_als, mf_als_model, movies, train_ratings, test_ratings, user_map, item_map, "als")
    save_results_pairwise(user_id, ranked_pop, ranked_pw_sgd, pairwise_model_sgd, mf_sgd_model, popularity_sgd, movies, train_ratings, user_map, item_map, "pw-sgd")
    save_results_pairwise(user_id, ranked_pop, ranked_pw_als, pairwise_model_als, mf_als_model, popularity_als, movies, train_ratings, user_map, item_map, "pw-als")
    print(f"[SAVE] Base ranker results saved")

def stage_mmr_reranking(
    user_id, ranked_pop, ranked_sgd, ranked_als, ranked_pw_sgd, ranked_pw_als,
    genre_matrix, item_to_idx, movies, train_ratings
):
    
    alphas = [0.1, 0.4, 0.7]
    base_rankers = [
        ("pop",    ranked_pop),
        ("sgd",    ranked_sgd),
        ("als",    ranked_als),
        ("pw-sgd", ranked_pw_sgd),
        ("pw-als", ranked_pw_als),
    ]
    
    for base_name, base_ranking in base_rankers:
        mmr_results = {
            alpha: mmr_rerank_from_ranking(
                base_ranking,
                genre_matrix,
                item_to_idx,
                alpha=alpha,
                top_M=MMR_TOP_M,
                output_K=10
            )
            for alpha in alphas
        }
        save_results_mmr(user_id, mmr_results, base_name, movies, train_ratings)
        print(f"[R5] MMR applied to {base_name}: alphas={alphas}")

def stage_personalization(
    train_ratings, test_ratings, mf_sgd_model, movies, user_map, item_map, all_items
): 
    relevant_map = build_relevant_items_map(test_ratings, threshold=4)
    target_users = [uid for uid in [1, 2, 3] if uid in user_map]

    if not target_users:
        print("[PERSONALIZATION] No target users available in current split.")
        return
    
    for p_user_id in target_users:
        print(f"\n[PERSONALIZATION] User {p_user_id}: Computing base SGD ranking...")
        p_candidates = get_candidates(p_user_id, train_ratings, all_items)
        p_ranked_sgd = rank_mf(p_candidates, p_user_id, mf_sgd_model, user_map, item_map)
        
        print(f"[PERSONALIZATION] User {p_user_id}:  EMA with 5 rounds...")
        personalization_summary = run_personalization_demo(
            user_id=p_user_id,
            base_ranking=p_ranked_sgd,
            movies=movies,
            mf_model=mf_sgd_model,
            user_map=user_map,
            item_map=item_map,
            out_dir="results/personalization",
            top_m=100,
            n_rounds=5,
            relevant_map=relevant_map,
        )
        
        print(f"[PERSONALIZATION] User {p_user_id} Summary:")
        for line in personalization_summary["discussion"]:
            print(f"  - {line}")

def main():

    session_log_path = "results/metrics/session_log.jsonl"
    if os.path.exists(session_log_path):
        os.remove(session_log_path)

    ratings, movies, train_ratings, test_ratings, user_ids, item_ids, user_map, item_map, all_items = load_data()

    if FAST_MODE:
        fast_users = set(user_ids[:FAST_USER_SUBSET])
        train_ratings = train_ratings[train_ratings["user_id"].isin(fast_users)].copy()
        test_ratings = test_ratings[test_ratings["user_id"].isin(fast_users)].copy()
        user_ids = np.array(sorted(train_ratings["user_id"].unique()))
        item_ids = np.array(sorted(train_ratings["item_id"].unique()))
        user_map = {uid: i for i, uid in enumerate(user_ids)}
        item_map = {iid: i for i, iid in enumerate(item_ids)}
        all_items = sorted(train_ratings["item_id"].unique())
        print(f"[FAST_MODE] Training users limited to first {FAST_USER_SUBSET}. Active users: {len(user_ids)}")
    
    candidates_user = get_candidates(1, train_ratings, all_items)
    
    ranked_pop = stage_baseline(candidates_user, train_ratings, movies)
    
    mf_sgd_model, mf_als_model = stage_matrix_factorization(
        len(user_ids), len(item_ids), train_ratings, user_map, item_map
    )
    
    ranked_sgd = rank_mf(candidates_user, 1, mf_sgd_model, user_map, item_map)
    ranked_als = rank_mf(candidates_user, 1, mf_als_model, user_map, item_map)
    
    pairwise_model_sgd, popularity_sgd, pairwise_model_als, popularity_als = stage_pairwise_ltr(
        train_ratings, mf_sgd_model, mf_als_model, user_map, item_map
    )
    
    ranked_pw_sgd = rank_pairwise(candidates_user, 1, mf_sgd_model, pairwise_model_sgd, user_map, item_map, popularity_sgd)
    ranked_pw_als = rank_pairwise(candidates_user, 1, mf_als_model, pairwise_model_als, user_map, item_map, popularity_als)
    
    stage_save_base_rankings(
        1, ranked_pop, ranked_sgd, ranked_als, ranked_pw_sgd, ranked_pw_als,
        mf_sgd_model, mf_als_model, popularity_sgd, popularity_als,
        pairwise_model_sgd, pairwise_model_als,
        movies, train_ratings, test_ratings, user_map, item_map
    )
    
    genre_matrix, item_to_idx = build_genre_matrix(movies)
    
    stage_mmr_reranking(
        1, ranked_pop, ranked_sgd, ranked_als, ranked_pw_sgd, ranked_pw_als,
        genre_matrix, item_to_idx, movies, train_ratings
    )
    
    stage_personalization(
        train_ratings, test_ratings, mf_sgd_model, movies, user_map, item_map, all_items
    )


if __name__ == "__main__":
    main()

