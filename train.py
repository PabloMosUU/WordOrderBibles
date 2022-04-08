from data import TokenizedBible, SplitData


class TrainedModel:
    pass

def train_model(split_data: SplitData) -> TrainedModel:
    # Todo: challenge 3
    raise NotImplementedError()

def produce_trained_model(bible: TokenizedBible) -> TrainedModel:
    split_data = bible.split()
    model = train_model(split_data)
    return model
