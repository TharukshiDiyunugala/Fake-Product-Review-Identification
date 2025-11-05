import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path="../data/fake_reviews.csv"):
    df = pd.read_csv(path)
    df = df[['text_', 'label']].dropna()
    return df

def split_data(df, test_size=0.2):
    return train_test_split(df['text_'], df['label'], test_size=test_size, random_state=42)
