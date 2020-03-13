import torch
from pandas import DataFrame


class NameCategoricalDataLoader():
    def __init__(self, df: DataFrame, batch_sz: int, name_header: str = 'name', probs_header: str = 'probs'):
        categories = torch.FloatTensor(df[probs_header].tolist())
        self.distribution = categories
        self.data_frame = df[name_header]
        self.batch_sz = batch_sz
        self.name_hdr = name_header
        self.probs_hdr = probs_header

    def sample(self):
        samples = []

        for i in range(self.batch_sz):
            sample = torch.distributions.Categorical(self.distribution).sample()
            samples.append(self.data_frame.iloc[sample.item()])

        return samples
