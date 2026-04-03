import pandas as pd
import numpy as np

def prep_user_paths(df):
    """
    Groups touchpoints by user_id to form complete paths and extract conversion.
    Returns a dataframe with user_id, path (list of channels), path_timestamps, and conversion.
    """
    # Sort by user and time
    df = df.sort_values(['user_id', 'timestamp'])
    
    paths = []
    
    for user_id, group in df.groupby('user_id'):
        channels = group['channel'].tolist()
        timestamps = group['timestamp'].tolist()
        converted = group['conversion'].max() # 1 if they converted, else 0
        costs = group['cost'].sum()
        
        paths.append({
            'user_id': user_id,
            'path': channels,
            'timestamps': timestamps,
            'conversion': converted,
            'total_cost': costs
        })
        
    return pd.DataFrame(paths)

def calculate_heuristic_models(paths_df):
    """
    Calculates First-Touch, Last-Touch, Linear, and Time-Decay attribution.
    Returns a DataFrame with the total conversions attributed to each channel by each model.
    """
    channels = ['Google', 'Meta', 'Email', 'SEO']
    
    # Initialize results
    results = {c: {'First-Touch': 0.0, 'Last-Touch': 0.0, 'Linear': 0.0, 'Time-Decay': 0.0} for c in channels}
    
    for _, row in paths_df.iterrows():
        if row['conversion'] == 1:
            path = row['path']
            n = len(path)
            
            if n == 0:
                continue
                
            # First-Touch
            results[path[0]]['First-Touch'] += 1.0
            
            # Last-Touch
            results[path[-1]]['Last-Touch'] += 1.0
            
            # Linear and Time-Decay
            if n == 1:
                results[path[0]]['Linear'] += 1.0
                results[path[0]]['Time-Decay'] += 1.0
            else:
                # Linear
                weight_linear = 1.0 / n
                for c in path:
                    results[c]['Linear'] += weight_linear
                    
                # Time-Decay (half-life of 7 days based on timestamps)
                # But to simplify, we can do a positional time decay or actual timestamp decay
                # Let's use positional decay: weight = 2^(idx - config)
                # or actual timestamps relative to the conversion timestamp
                conv_time = row['timestamps'][-1]
                decay_weights = []
                for t in row['timestamps']:
                    # time diff in days
                    diff_days = (conv_time - t).days + (conv_time - t).seconds / 86400.0
                    decay_weights.append(np.exp(-0.1 * diff_days)) # arbitrary decay factor
                
                sum_weights = sum(decay_weights)
                
                for c, w in zip(path, decay_weights):
                    results[c]['Time-Decay'] += (w / sum_weights)
                    
    # Convert to DataFrame
    df_res = pd.DataFrame(results).T.reset_index().rename(columns={'index': 'Channel'})
    return df_res
