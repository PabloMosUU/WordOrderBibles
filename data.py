class SplitData:
    pass

class Bible:
    def split(self) -> SplitData:
        # Todo: challenge 2
        raise NotImplementedError()
    pass

def parse_file(filename: str) -> Bible:
    raise NotImplementedError()

def preprocess(bible: Bible) -> Bible:
    # Todo: challenge 1
    # Include eventual verse filters
    raise NotImplementedError()

def process_bible(filename: str) -> Bible:
    structured_bible = parse_file(filename)
    return preprocess(structured_bible)
