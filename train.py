from data import TokenizedBible, SplitData


class TrainedModel:
    pass

def train_model(split_data: SplitData) -> TrainedModel:
    # Todo: challenge 3
    raise NotImplementedError()

def produce_trained_model(bible: TokenizedBible,
                          hold_out_fraction: float,
                          test_fraction: float) -> TrainedModel:
    split_data = bible.split(hold_out_fraction, test_fraction)
    model = train_model(split_data)
    return model
