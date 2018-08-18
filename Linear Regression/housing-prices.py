import sklearn.datasets as skds
import pandas as pd
dataset = skds.load_boston()

df = pd.DataFrame(dataset.data)
df.columns = dataset['feature_names']
df['PRICE'] = dataset.target*1000

print(df.head())
