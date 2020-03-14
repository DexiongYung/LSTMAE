import argparse
import pandas as pd
from pandas import DataFrame

parser = argparse.ArgumentParser()
parser.add_argument('--file_pth', help='Path of file', nargs='?', type=str)
parser.add_argument('--save_pth', help='Path of the save csv', nargs='?', type=str)

args = parser.parse_args()
df = pd.read_csv(args.file_pth)

sum = df['count'].sum()

probs = []

for idx, row in df.iterrows():
    count = row['count']
    probability = count / sum
    probs.append(probability)

df['probs'] = probs

df.to_csv(args.save_pth)
