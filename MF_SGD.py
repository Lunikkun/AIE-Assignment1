import numpy as np
import os

class MF_SGD:
    def __init__(self, n_users, n_items, n_factors=20, reg=0.05, lr=0.01, patience=2):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.reg = reg 
        self.lr = lr     
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
    def train(self, train_ratings, user_map, item_map, epochs=20, val_split=0.1):
    
        train_data = train_ratings.sample(frac=1 - val_split, random_state=42)
        val_data = train_ratings.drop(train_data.index)
        
        self.mu = train_data['rating'].mean()
        
        for epoch in range(epochs):
            total_loss = 0
            for _, row in train_data.iterrows():
                u = user_map[row['user_id']]
                i = item_map[row['item_id']]
                r = row['rating']
                
                pred = self.predict(u, i)
                err = r - pred
                
                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])

                p_u_old = self.P[u].copy()
                self.P[u] += self.lr * (err * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (err * p_u_old - self.reg * self.Q[i])
                
                total_loss += err ** 2
            
            train_loss = total_loss / len(train_data)
            
            val_loss = 0
            for _, row in val_data.iterrows():
                u = user_map[row['user_id']]
                i = item_map[row['item_id']]
                r = row['rating']
                pred = self.predict(u, i)
                val_loss += (r - pred) ** 2
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
            lr=self.lr,
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
            lr=float(data["lr"]),
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