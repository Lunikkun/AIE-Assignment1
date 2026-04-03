import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def build_genre_matrix(movies_df):

    genre_cols = [f'genre_{i}' for i in range(19)]
    genre_vectors = movies_df[genre_cols].values.astype(float)

    norms = np.linalg.norm(genre_vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    genre_matrix = genre_vectors / norms

    item_to_idx = {int(mid): i for i, mid in enumerate(movies_df['movie_id'])}

    return genre_matrix, item_to_idx


def mmr_rerank(candidates, rel_scores, genre_matrix, item_to_idx, alpha, output_K=10):

    valid = [c for c in candidates if c in item_to_idx]

    selected = []       
    remaining = list(valid)  

    while remaining and len(selected) < output_K:

        if not selected:
            best = max(remaining, key=lambda c: rel_scores.get(c, 0.0))
        else:
            selected_vecs = genre_matrix[[item_to_idx[s] for s in selected]]

            best, best_score = None, -np.inf
            for c in remaining:
                c_vec = genre_matrix[item_to_idx[c]].reshape(1, -1)

                sims = cosine_similarity(c_vec, selected_vecs)[0]
                max_sim = float(sims.max())

                score = (1.0 - alpha) * rel_scores.get(c, 0.0) - alpha * max_sim

                if score > best_score:
                    best_score = score
                    best = c

        selected.append(best)
        remaining.remove(best)

    return selected


def mmr_rerank_from_ranking(ranked_candidates, genre_matrix, item_to_idx,
                            alpha, top_M=50, output_K=10):

    top_candidates = [c for c in ranked_candidates if c in item_to_idx][:top_M]
    n = len(top_candidates)

    rel_scores = {c: 1.0 - (i / n) for i, c in enumerate(top_candidates)}

    return mmr_rerank(top_candidates, rel_scores, genre_matrix, item_to_idx,
                      alpha=alpha, output_K=output_K)
