from pandas import DataFrame
from torch.utils.data import Dataset


class NameDataset(Dataset):
    def __init__(self, df: DataFrame, col_name: str):
        df = df[df['count'] > 5000]
        self.data_frame = df[col_name].str.lower().dropna()

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        return self.data_frame.iloc[index]
