import numpy as np
import os

class MF_ALS:
    def __init__(self, n_users, n_items, n_factors=20, reg=0.05, patience=2):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.reg = reg
        self.patience = patience  
        
        self.P = np.random.normal(0, 0.1, (n_users, n_factors))  
        self.Q = np.random.normal(0, 0.1, (n_items, n_factors)) 
        
        self.bu = np.zeros(n_users)
        self.bi = np.zeros(n_items)
        self.mu = 0
        
        self.best_loss = float('inf')
        self.early_stop_counter = 0

    def predict(self, u, i):
        score = self.mu + self.bu[u] + self.bi[i] + np.dot(self.P[u], self.Q[i])
        return score
    
    def train(self, train_ratings, user_map, item_map, epochs=10, val_split=0.1):

        train_data = train_ratings.sample(frac=1 - val_split, random_state=42)
        val_data = train_ratings.drop(train_data.index)
        
        self.mu = train_data['rating'].mean()
        
        user_ids_list = [None] * len(user_map)
        for uid, idx in user_map.items():
            user_ids_list[idx] = uid
        item_ids_list = [None] * len(item_map)
        for iid, idx in item_map.items():
            item_ids_list[idx] = iid
        
        for epoch in range(epochs):
            for i in range(self.n_items):
                users_rated = train_data[train_data['item_id'] == item_ids_list[i]]
                if not users_rated.empty:
                    u_indices = [user_map[uid] for uid in users_rated['user_id']]
                    ratings_vec = users_rated['rating'].values - self.mu - self.bu[u_indices] - self.bi[i]
                    P_u = self.P[u_indices]
                    self.Q[i] = np.linalg.solve(P_u.T @ P_u + self.reg * np.eye(self.n_factors), P_u.T @ ratings_vec)
                    self.bi[i] = np.mean(users_rated['rating'].values - self.mu - self.bu[u_indices] - P_u @ self.Q[i])

            for u in range(self.n_users):
                items_rated = train_data[train_data['user_id'] == user_ids_list[u]]
                if not items_rated.empty:
                    i_indices = [item_map[iid] for iid in items_rated['item_id']]
                    ratings_vec = items_rated['rating'].values - self.mu - self.bu[u] - self.bi[i_indices]
                    Q_i = self.Q[i_indices]
                    self.P[u] = np.linalg.solve(Q_i.T @ Q_i + self.reg * np.eye(self.n_factors), Q_i.T @ ratings_vec)
                    self.bu[u] = np.mean(items_rated['rating'].values - self.mu - self.bi[i_indices] - Q_i @ self.P[u])
            
            total_loss = 0
            for _, row in train_data.iterrows():
                u = user_map[row['user_id']]
                i = item_map[row['item_id']]
                pred = self.predict(u, i)
                total_loss += (row['rating'] - pred) ** 2
            train_loss = total_loss / len(train_data)
            
            val_loss = 0
            for _, row in val_data.iterrows():
                u = user_map[row['user_id']]
                i = item_map[row['item_id']]
                pred = self.predict(u, i)
                val_loss += (row['rating'] - pred) ** 2
            val_loss /= len(val_data)
            
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(
            path,
            n_users=self.n_users,
            n_items=self.n_items,
            n_factors=self.n_factors,
            reg=self.reg,
            patience=self.patience,
            mu=self.mu,
            best_loss=self.best_loss,
            early_stop_counter=self.early_stop_counter,
            P=self.P,
            Q=self.Q,
            bu=self.bu,
            bi=self.bi,
        )

    @classmethod
    def load(cls, path):
        data = np.load(path)
        model = cls(
            int(data["n_users"]),
            int(data["n_items"]),
            n_factors=int(data["n_factors"]),
            reg=float(data["reg"]),
            patience=int(data["patience"]),
        )
        model.mu = float(data["mu"])
        model.best_loss = float(data["best_loss"])
        model.early_stop_counter = int(data["early_stop_counter"])
        model.P = data["P"]
        model.Q = data["Q"]
        model.bu = data["bu"]
        model.bi = data["bi"]
        return model