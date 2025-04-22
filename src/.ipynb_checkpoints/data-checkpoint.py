import pandas as pd


def load_data(train_path: str, test_path: str, sample_path: str):
    """Load train, test, and sample submission."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)
    return train, test, sample