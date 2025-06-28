import math
from collections import defaultdict, Counter


class TFIDF_Calc:    
    def __init__(self, on: str):
        """
        Args:
            on (str): The key corresponding to the text field in the input dictionary.
        """
        self.on = on
        
        # State-tracking variables for online learning
        self.doc_count = 0
        self.doc_freqs = {}  # Stores document frequency for each term
        self.vocabulary = {} # Maps each term to a unique feature index
        self._next_vocab_idx = 0

    def _tokenize(self, text: str):
        """A simple tokenizer. A more advanced version could use regex or a library."""
        return text.lower().split()

    def learn_one(self, x: dict):
        """
        Updates the internal state based on a single document.
        This includes document count, document frequencies, and the vocabulary.
        """
        text = x.get(self.on, "")
        tokens = self._tokenize(text)

        # Increment total document count
        self.doc_count += 1
        
        # Update document frequencies for unique tokens in this document
        for token in set(tokens):
            self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
            # Add new tokens to the vocabulary
            if token not in self.vocabulary:
                self.vocabulary[token] = self._next_vocab_idx
                self._next_vocab_idx += 1
        
        return self

    def transform_one(self, x: dict):
        """
        Transforms a single document into a TF-IDF sparse vector.
        """
        text = x.get(self.on, "")
        tokens = self._tokenize(text)
        
        if not tokens:
            return {}

        # 1. Calculate Term Frequencies (TF) for the current document
        token_counts = Counter(tokens)
        total_tokens_in_doc = len(tokens)
        
        tfidf_vector = {}
        
        for token, count in token_counts.items():
            # Only generate features for words we have learned in the vocabulary
            if token in self.vocabulary:
                # Term Frequency calculation
                tf = count / total_tokens_in_doc
                
                # 2. Calculate Inverse Document Frequency (IDF) using the global state
                # We use a standard smoothed IDF formula to prevent division by zero
                # and to moderate the weight of rare words.
                # Formula: log((N+1) / (df+1)) + 1
                # N = total documents seen, df = documents containing the term
                df = self.doc_freqs.get(token, 0)
                idf = math.log((self.doc_count + 1) / (df + 1)) + 1
                
                # 3. Combine to get TF-IDF score
                feature_index = self.vocabulary[token]
                tfidf_vector[feature_index] = tf * idf
        
        return tfidf_vector
        
    def clone(self, include_attributes=True):
        return TFIDF_Calc(on=self.on)
        
    def _get_text(self, x):
        return x.get(self.on, "") if self.on is not None else x
        
    def _tokenize(self, text):
        if isinstance(text, str):
            return text.lower().split()
        return [str(text).lower()]
