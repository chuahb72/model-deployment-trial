import pandas as pd

# Creating a simple dataset: Area (sq ft) vs Price Category
data = {
    'sqft': [500, 700, 800, 1200, 1500, 1800, 2500, 3000, 3500, 4000],
    'is_expensive': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1] # 0 = Cheap, 1 = Expensive
}

df = pd.DataFrame(data)
df.to_csv('data.csv', index=False)
print("Dataset 'data.csv' created!")