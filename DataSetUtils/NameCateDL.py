from pandas import DataFrame

class NameCategoricalDataLoader():
    def __init__(self, df: DataFrame, col_name: str, batch_sz: int):        
        df = df[df['count'] > 5000]
        self.data_frame = df[col_name].str.lower().dropna()
        self.batch_sz = batch_sz

