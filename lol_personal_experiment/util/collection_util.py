import pickle


def save(obj, filename, compressed=False):
    """ obj를 pickle 파일(.pkl)로 저장 """
    if compressed:
        import gzip
        with gzip.open(filename, 'wb') as f:
            pickle.dump(obj, f)
    else:
        with open(filename, 'wb') as f:
            pickle.dump(obj, f)


def load(filename, compressed=False):
    """ pickle 파일(.pkl)로부터 로드한 데이터를 반환 """
    if compressed:
        import gzip
        with gzip.open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        with open(filename, 'rb') as f:
            return pickle.load(f)
