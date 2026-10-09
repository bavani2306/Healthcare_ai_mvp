import pandas as pd
import re

class Preprocessor:
    def transform(self, data: dict):
        return pd.DataFrame([data])
    def preprocess_document(self, text: str):
        text = re.sub(r'[^a-zA-Z0-9\s.,]', '', text)
        return text.lower().strip()