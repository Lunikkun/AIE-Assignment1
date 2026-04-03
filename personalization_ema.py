import json
import os
import numpy as np

from metrics import build_relevant_items_map, recall_at_k, ndcg_at_k_binary, diversity_at_k
from mmr import build_genre_matrix


GENRE_COLS = [f"genre_{i}" for i in range(19)]


def _normalize(vec):
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return vec / norm


def _get_title(movies, item_id):
    rows = movies[movies["movie_id"] == item_id]["movie_title"].values
    return rows[0] if len(rows) > 0 else "N/A"


def _build_item_vectors_from_mf(mf_model, item_map):
    vectors = {}
    for item_id, idx in item_map.items():
        vectors[item_id] = _normalize(mf_model.Q[idx].astype(float))
    return vectors


def _build_item_vectors_from_genre(movies):
    vectors = {}
    for row in movies[["movie_id"] + GENRE_COLS].itertuples(index=False):
        item_id = int(row.movie_id)
        v = np.array(row[1:], dtype=float)
        vectors[item_id] = _normalize(v)
    return vectors


def _prepare_candidates(base_ranking, item_vectors, top_m):
    return [item for item in base_ranking if item in item_vectors][:top_m]


def _score_candidates(candidates, user_state, item_vectors):
    scored = [(item, float(np.dot(user_state, item_vectors[item]))) for item in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _choose_item_heuristic(slate, already_chosen):
    for item_id, _ in slate:
        if item_id not in already_chosen:
            return item_id
    return slate[0][0]


def _ema_update(user_state, item_vector, rho):
    updated = (1.0 - rho) * user_state + rho * item_vector
    return _normalize(updated)


def _state_summary(user_state):
    top_dims = np.argsort(np.abs(user_state))[-3:][::-1]
    return {
        "l2_norm": float(np.linalg.norm(user_state)),
        "top_dimensions": [
            {"index": int(idx), "value": float(user_state[idx])}
            for idx in top_dims
        ],
    }


def _simulate_one_session(user_id, session_name, base_candidates, item_vectors, init_user_state,
                          movies, rho, n_rounds=5, slate_size=10,
                          user_map=None, item_map=None, relevant_items=None,
                          genre_matrix=None, item_to_idx=None, method_name=None):
    user_state = _normalize(init_user_state.copy())
    chosen = set()
    rounds = []
    jsonl_records = []

    for r in range(1, n_rounds + 1):
        scored = _score_candidates(base_candidates, user_state, item_vectors)
        slate = scored[:slate_size]

        picked_item = _choose_item_heuristic(slate, chosen)
        picked_score = next(score for item, score in slate if item == picked_item)
        picked_title = _get_title(movies, picked_item)

        prev_state = user_state.copy()
        user_state = _ema_update(user_state, item_vectors[picked_item], rho)
        drift = float(np.linalg.norm(user_state - prev_state))

        rounds.append({
            "round": r,
            "slate_top10": [
                {
                    "item_id": int(item),
                    "title": _get_title(movies, item),
                    "personalized_score": float(score),
                }
                for item, score in slate
            ],
            "picked_item": {
                "item_id": int(picked_item),
                "title": picked_title,
                "score_at_pick": float(picked_score),
            },
            "state_drift_l2": drift,
        })

        if relevant_items is not None and user_map is not None and item_map is not None:
            recommended_ids = [int(item) for item, _ in slate]
            jsonl_records.append({
                "record_type": "personalization_session",
                "user_internal_idx": int(user_map[int(user_id)]),
                "user_id": int(user_id),
                "method": method_name or "EMA",
                "method_name": method_name or "EMA",
                "session_name": session_name,
                "round": int(r),
                "top_k": [int(item_map[item_id]) for item_id in recommended_ids],
                "top_k_item_ids": recommended_ids,
                "metrics": {
                    "Recall@10": float(recall_at_k(recommended_ids, relevant_items, slate_size) or 0.0),
                    "NDCG@10": float(ndcg_at_k_binary(recommended_ids, relevant_items, slate_size) or 0.0),
                    "Diversity@10": float(diversity_at_k(recommended_ids, genre_matrix, item_to_idx, slate_size)),
                },
                "hyperparameters": {
                    "rho": float(rho),
                    "K": int(slate_size),
                    "representation": session_name.split("_")[2].upper(),
                },
                "chosen_item": {
                    "internal_idx": int(item_map[int(picked_item)]),
                    "item_id": int(picked_item),
                    "title": picked_title,
                    "score_at_pick": float(picked_score),
                },
                "updated_state_summary": _state_summary(user_state),
            })

        chosen.add(picked_item)

    first_ids = [x["item_id"] for x in rounds[0]["slate_top10"]]
    last_ids = [x["item_id"] for x in rounds[-1]["slate_top10"]]
    overlap = len(set(first_ids).intersection(last_ids))

    session_summary = {
        "session_name": session_name,
        "rho": float(rho),
        "n_rounds": int(n_rounds),
        "first_vs_last_slate_overlap_top10": int(overlap),
        "picked_sequence": [step["picked_item"]["title"] for step in rounds],
        "avg_state_drift_l2": float(np.mean([step["state_drift_l2"] for step in rounds])),
    }

    return {
        "user_id": int(user_id),
        "session_name": session_name,
        "rho": float(rho),
        "update_rule": "u_{t+1} = normalize((1-rho)u_t + rho v_i)",
        "rounds": rounds,
        "summary": session_summary,
        "jsonl_records": jsonl_records,
    }


def run_personalization_demo(user_id, base_ranking, movies, mf_model, user_map, item_map,
                             out_dir="results/personalization", top_m=100, n_rounds=5,
                             relevant_map=None):
    user_out_dir = os.path.join(out_dir, f"user_{user_id}")
    os.makedirs(user_out_dir, exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)

    mf_vectors = _build_item_vectors_from_mf(mf_model, item_map)
    genre_vectors = _build_item_vectors_from_genre(movies)
    genre_matrix, item_to_idx = build_genre_matrix(movies)

    mf_candidates = _prepare_candidates(base_ranking, mf_vectors, top_m=top_m)
    genre_candidates = _prepare_candidates(base_ranking, genre_vectors, top_m=top_m)

    u_idx = user_map[user_id]
    init_mf_state = _normalize(mf_model.P[u_idx].astype(float))

    if len(genre_candidates) == 0:
        init_genre_state = np.zeros(len(GENRE_COLS), dtype=float)
    else:
        init_genre_state = _normalize(np.mean([genre_vectors[i] for i in genre_candidates[:10]], axis=0))

    sessions = [
        {
            "name": "session_1_mf_rho_0.1",
            "rho": 0.1,
            "vectors": mf_vectors,
            "candidates": mf_candidates,
            "init_state": init_mf_state,
        },
        {
            "name": "session_2_mf_rho_0.3",
            "rho": 0.3,
            "vectors": mf_vectors,
            "candidates": mf_candidates,
            "init_state": init_mf_state,
        },
        {
            "name": "session_3_genre_rho_0.3",
            "rho": 0.3,
            "vectors": genre_vectors,
            "candidates": genre_candidates,
            "init_state": init_genre_state,
        },
    ]

    session_outputs = []
    session_log_records = []
    for sess in sessions:
        output = _simulate_one_session(
            user_id=user_id,
            session_name=sess["name"],
            base_candidates=sess["candidates"],
            item_vectors=sess["vectors"],
            init_user_state=sess["init_state"],
            movies=movies,
            rho=sess["rho"],
            n_rounds=n_rounds,
            slate_size=10,
            user_map=user_map,
            item_map=item_map,
            relevant_items=(relevant_map or {}).get(int(user_id), set()),
            genre_matrix=genre_matrix,
            item_to_idx=item_to_idx,
            method_name="EMA",
        )
        session_outputs.append(output)
        session_log_records.extend(output["jsonl_records"])

        file_path = os.path.join(user_out_dir, f"{sess['name']}.json")
        with open(file_path, "w") as f:
            json.dump(output, f, indent=4)

    session_jsonl_path = os.path.join("results", "metrics", "session_log.jsonl")
    with open(session_jsonl_path, "a") as f:
        for record in session_log_records:
            f.write(json.dumps(record) + "\n")

    discussion = []
    for out in session_outputs:
        overlap = out["summary"]["first_vs_last_slate_overlap_top10"]
        rho = out["summary"]["rho"]
        drift = out["summary"]["avg_state_drift_l2"]
        discussion.append(
            f"{out['session_name']}: overlap top10 round1->round5 = {overlap}/10, "
            f"rho={rho}, drift medio stato={drift:.4f}."
        )

    summary = {
        "user_id": int(user_id),
        "requirement_check": {
            "sessions": 3,
            "rounds_per_session": int(n_rounds),
            "update_rule": "u_{t+1} = normalize((1-rho)u_t + rho v_i)",
            "item_representation_used": ["MF q_i", "Genre vector"],
        },
        "discussion": discussion,
        "session_files": [f"{x['session_name']}.json" for x in session_outputs],
    }

    summary_path = os.path.join(user_out_dir, f"summary_user{user_id}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    return summary


def run_personalization_demo_genre_only(user_id, base_ranking, movies,
                                        out_dir="results/personalization/genre_only",
                                        top_m=100, n_rounds=5):
    user_out_dir = os.path.join(out_dir, f"user_{user_id}")
    os.makedirs(user_out_dir, exist_ok=True)

    genre_vectors = _build_item_vectors_from_genre(movies)
    genre_candidates = _prepare_candidates(base_ranking, genre_vectors, top_m=top_m)

    if len(genre_candidates) == 0:
        init_genre_state = np.zeros(len(GENRE_COLS), dtype=float)
    else:
        init_genre_state = _normalize(np.mean([genre_vectors[i] for i in genre_candidates[:10]], axis=0))

    sessions = [
        {"name": "session_1_genre_rho_0.1", "rho": 0.1},
        {"name": "session_2_genre_rho_0.4", "rho": 0.4},
        {"name": "session_3_genre_rho_0.7", "rho": 0.7},
    ]

    session_outputs = []
    for sess in sessions:
        output = _simulate_one_session(
            user_id=user_id,
            session_name=sess["name"],
            base_candidates=genre_candidates,
            item_vectors=genre_vectors,
            init_user_state=init_genre_state,
            movies=movies,
            rho=sess["rho"],
            n_rounds=n_rounds,
            slate_size=10,
        )
        session_outputs.append(output)

        file_path = os.path.join(user_out_dir, f"{sess['name']}.json")
        with open(file_path, "w") as f:
            json.dump(output, f, indent=4)

    discussion = []
    for out in session_outputs:
        overlap = out["summary"]["first_vs_last_slate_overlap_top10"]
        rho = out["summary"]["rho"]
        drift = out["summary"]["avg_state_drift_l2"]
        discussion.append(
            f"{out['session_name']}: overlap top10 round1->round5 = {overlap}/10, "
            f"rho={rho}, drift medio stato={drift:.4f}."
        )

    summary = {
        "user_id": int(user_id),
        "requirement_check": {
            "sessions": 3,
            "rounds_per_session": int(n_rounds),
            "update_rule": "u_{t+1} = normalize((1-rho)u_t + rho v_i)",
            "item_representation_used": ["Genre vector"],
        },
        "discussion": discussion,
        "session_files": [f"{x['session_name']}.json" for x in session_outputs],
    }

    summary_path = os.path.join(user_out_dir, f"summary_user{user_id}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    return summary
