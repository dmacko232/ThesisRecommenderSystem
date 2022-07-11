from typing import List, Dict
import math
import numpy as np

from collections import defaultdict

class BM25:
    """
    Own implementation of BM25, specifically BM25+ see http://www.cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf

    Implementation takes tokenized documents, processes them and then can be used to get score for
    whole query or even for each token (to build matrix).

    Attributes
    ------------
    No public attributes
    """

    def __init__(self, tokenized_docs: List[List[str]], k1: float=1.5, b: float=0.75, delta: float=0.5) -> "None":
        """
        Initializes BM25Plus model with specified documents and parameters.
        
        Parameters
        -----------
        tokenized_docs: List[List[str]]
            corpus of documents which are tokenized and ready to be used
        k1: float
            BM25+ parameter, for more info see the paper
        b: float
            BM25+ parameter, for more info see the paper
        delta: float
            BM25+ parameter, for more info see the paper
        """

        # calculate stats about docs 
        doc_freqs = [] # frequency dict for each doc
        doc_lens = [] # lens of each doc
        token_freqs_per_doc = defaultdict(lambda: 0) # token occurences per doc (for example 1 of word is in one doc)
        total_tokens = 0 # total tokens in whole doc
        for doc in tokenized_docs:
            total_tokens += len(doc)
            doc_lens.append(len(doc))

            freqs = defaultdict(lambda: 0)
            # add each token to frequency dict
            for token in doc:
                freqs[token] += 1
            doc_freqs.append(freqs)

            # add each token that 
            for token in freqs.keys():
                token_freqs_per_doc[token] += 1

        # calculate inverse document frequencies
        docs_count = len(tokenized_docs)
        idfs = defaultdict(lambda: 0, {
            token: math.log(docs_count + 1) - math.log(freq + 0.5)
            for token, freq
            in token_freqs_per_doc.items()
        })

        # store all the needed attributes
        self._idfs = idfs # inverse document frequencies
        self._doc_freqs = doc_freqs # token frequencies for each doc
        self._doc_lens = doc_lens # lengths of documents
        self._docs_count = docs_count # how many docs are there (n)
        self._avg_doc_len = total_tokens / self._docs_count
        # below are parameters for the calculation
        self._k1 = k1
        self._b = b
        self._delta = delta

    # assign to each document one number corresponding to score from tokenized_doc
    def score(self, tokenized_doc: List[str]) -> List[float]:
        """
        Scores each document that was the model initalized with to the supplied one here (query)

        Parameters
        ----------
        tokenized_doc: List[str]
            tokenized document (also called query) to score all the documents to
        
        Returns
        --------
        List[float]
            list of scores (bigger is more similar) for each document
        """
        
        return [sum(score_dict.values()) for score_dict in self.score_tokenwise(tokenized_doc)]

    # get dictionary of token scores, pretty much transform that creates matrix
    def score_tokenwise(self, tokenized_doc: List[str]) -> List[Dict[str, float]]:
        """
        Scores each document's tokens that was the model initalized with to the supplied one here (query)

        Parameters
        ----------
        tokenized_doc: List[str]
            tokenized document (also called query) to score all the documents to
        
        Returns
        --------
        List[Dict[str, float]]
            list of dictionaries where each dictionary has scores for all tokens in the doc (result score is sum)
        """
        
        scores = []
        for i in range(self._docs_count):
            scores.append(
                {t: self._score_token_for_doc(t, i) for t in tokenized_doc}
                )
        return scores

    # scores one token for all the documents
    def _score_token(self, token: str) -> List[float]:

        scores = []
        for i in range(self._docs_count):
            scores.append(self._score_token_for_doc(token, i))
        return scores

    # takes one token and document specified by given index and scores it accordingly
    def _score_token_for_doc(self, token: str, doc_index: int) -> float:

        token_freq = self._doc_freqs[doc_index][token]
        token_idf = self._idfs[token]
        doc_len = self._doc_lens[doc_index]
        # see 3.2 BM25L in http://www.cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf
        #ctd = token_freq / (1 - self._b + self._b * doc_len / self._avg_doc_len)
        #token_score = token_idf * (((self._k1 + 1) * (ctd + self._delta)) / (self._k1 + ctd + self._delta))

        # see 3.3 BM25Plus
        token_score = token_idf * (self._delta + (
            token_freq * (self._k1 + 1)) / ( # nominator
                self._k1 * (1 - self._b + self._b * doc_len / self._avg_doc_len) + token_freq) # denominator
            )
        return token_score
