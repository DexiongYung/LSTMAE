from pandas import DataFrame
from torch.utils.data import Dataset


class NameDataset(Dataset):
    def __init__(self, df: DataFrame, col_name: str):
        self.data_frame = df[col_name].apply(lambda x: ''.join([" " if ord(i) < 32 or ord(i) > 126 else i for i in x]))

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        return self.data_frame.iloc[index]
