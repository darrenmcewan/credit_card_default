import pandas as pd
import numpy as np

df = pd.read_csv('creditCardDefaults.csv')
df.head(10)

print(df.isnull().sum())        
print(df.isnull().sum().sum())
