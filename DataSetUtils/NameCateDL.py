import torch
from pandas import DataFrame


class NameCategoricalDataLoader():
    def __init__(self, df: DataFrame, batch_sz: int, name_header: str = 'name', count_header: str = 'count'):
        self.name_hdr = name_header
        self.count_hdr = count_header
        self.data_frame = df[name_header].dropna()
        self.batch_sz = batch_sz

        categories = []
        count_sum = 0

        for idx, row in df.iterrows():
            count = row[self.count_hdr]
            categories.append(count)
            count_sum += count

        categories = torch.FloatTensor(categories)
        self.distribution = categories * (1 / count_sum)

    def sample(self):
        samples = []

        for i in range(self.batch_sz):
            sample = torch.distributions.Categorical(self.distribution).sample()
            samples.append(self.data_frame.iloc[sample.item()])

        return samples
