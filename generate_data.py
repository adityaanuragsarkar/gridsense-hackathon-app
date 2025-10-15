# File: generate_data.py
import pandas as pd
import numpy as np
NUM_ROWS = 10000
NODES = ['Substation', 'Hospital', 'Factory', 'DataCenter', 'Homes']
data = []
for i in range(NUM_ROWS):
    hour = i % 24
    node = np.random.choice(NODES)
    load = np.random.randint(50, 80) if 8 <= hour < 18 else np.random.randint(20, 50)
    temp = np.random.uniform(25.0, 45.0) if 10 <= hour < 16 else np.random.uniform(15.0, 25.0)
    storm_active = np.random.choice([0, 1], p=[0.95, 0.05])
    failure_chance = 0.01
    if load > 75: failure_chance += 0.1
    if temp > 40: failure_chance += 0.1
    if storm_active: failure_chance += 0.3
    failed_in_next_hour = 1 if np.random.rand() < failure_chance else 0
    data.append([hour, NODES.index(node), load, temp, storm_active, failed_in_next_hour])
columns = ['hour', 'node_id', 'load', 'temp', 'storm_active', 'failed_in_next_hour']
df = pd.DataFrame(data, columns=columns)
df.to_csv('grid_data.csv', index=False)
print("✅ Created grid_data.csv")
