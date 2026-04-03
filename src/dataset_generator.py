import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_synthetic_data(num_users=10000, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    
    channels = ['Google', 'Meta', 'Email', 'SEO']
    
    # Base configuration for channels
    # Each has a base conversion prob and average cost per touchpoint
    channel_config = {
        'Google': {'base_conv': 0.05, 'cost_mean': 1.5, 'cost_std': 0.5},
        'Meta': {'base_conv': 0.04, 'cost_mean': 1.2, 'cost_std': 0.4},
        'Email': {'base_conv': 0.08, 'cost_mean': 0.1, 'cost_std': 0.05},
        'SEO': {'base_conv': 0.06, 'cost_mean': 0.0, 'cost_std': 0.0} # Organic is "free" per click usually, but we can assign arbitrary 0 cost here
    }
    
    data = []
    
    start_date = datetime(2025, 1, 1)
    
    for i in range(num_users):
        user_id = f"U{i:06d}"
        
        # Decide how many touchpoints this user will have (1 to 5)
        num_touchpoints = min(5, int(np.random.exponential(1.5)) + 1)
        
        user_path = []
        user_cost = 0.0
        
        # Generate touchpoints
        current_time = start_date + timedelta(days=random.randint(0, 30), hours=random.randint(0, 23))
        
        for t_idx in range(num_touchpoints):
            # Pick a channel. Maybe slightly biased toward Meta/Google early on, Email later?
            channel = random.choices(
                channels,
                weights=[0.4, 0.4, 0.1, 0.1] if t_idx == 0 else [0.2, 0.2, 0.4, 0.2],
                k=1
            )[0]
            
            # Calculate cost for this touchpoint
            conf = channel_config[channel]
            cost = max(0, np.random.normal(conf['cost_mean'], conf['cost_std']))
            user_cost += cost
            
            user_path.append(channel)
            
            # Record touchpoint
            data.append({
                'user_id': user_id,
                'timestamp': current_time,
                'channel': channel,
                'cost': cost,
                'touchpoint_index': t_idx + 1,
                'is_last': (t_idx == num_touchpoints - 1)
            })
            
            current_time += timedelta(days=random.randint(0, 5), hours=random.randint(1, 24))

        # Determine conversion based on the path
        # A synergistic effect: if both Google and Email are in the path, higher probability
        base_prob = 0.01 + len(set(user_path)) * 0.015 # More unique channels = higher chance
        
        if user_path[-1] == 'Email':
            base_prob += 0.03
        if 'Google' in user_path and 'Meta' in user_path:
            base_prob += 0.04
            
        conversion = 1 if random.random() < base_prob else 0
        
        # Mark conversion on the LAST touchpoint only
        data[-1]['conversion'] = conversion

    df = pd.DataFrame(data)
    
    # Fill non-last touchpoints with conversion = 0
    df['conversion'] = df['conversion'].fillna(0).astype(int)
    
    # Save to CSV
    df.to_csv("synthetic_data.csv", index=False)
    print(f"Generated synthetic_data.csv with {len(df)} touchpoints from {num_users} users.")
    
    return df

if __name__ == "__main__":
    generate_synthetic_data()
