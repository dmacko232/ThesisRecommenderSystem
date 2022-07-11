import abc
from typing import List, Set, Dict, Optional, Tuple, Iterable, Any
from dataclasses import dataclass
from encoders import OneHotEncoder
from similarities import jaccard_distance, cosine_similarity

from load_data import Thesis
from bm25 import BM25
from preprocess import (
    clean_theses, 
    preprocess_keywords, 
    preprocess_cs_abstract, 
    preprocess_en_abstract, 
    preprocess_description, 
    preprocess_topic,
    extract_docs_stopwords
    )

# INTERFACES

class ThesisRecommenderSystemBase(abc.ABC):
    """Thesis Recommender System Base."""

    @abc.abstractmethod
    def recommend(self, thesis: Thesis, n_recommendations: int=10) -> List[Tuple[int, Thesis]]:
        """
        Recommend theses that are most similar.

        Parameters
        ----------
        thesis: Thesis
            Valid Thesis object whose similar objects are needed.
        n_recommendations: int
            Positive number that specifies the amount of recommendations needed.
        
        Return
        ----------
        List[Tuple[int, Thesis]]
            List of `n_recommendations` of (index, Thesis) where the Thesis is are the most similar to `thesis`.
        """

        pass

class ThesisScorerBase(abc.ABC):
    """Thesis Scorer Base."""

    @abc.abstractmethod
    def score(self, thesis: Thesis) -> List[float]:
        """
        Scores all the theses according to the supplied `thesis`

        Parameters
        ----------
        thesis: Thesis
            Valid Thesis object whose similar objects are needed.
        
        Return
        ----------
        List[float]
            List of scores for each of the `theses` supplied in constructor to `thesis`.

        Note
            Bigger score is better here!
        """

        pass

# SCORER CLASSES

class ThesisScorerKeywords(ThesisScorerBase):
    """Thesis Scorer that only uses Jaccard Score for keywords."""

    def __init__(self, theses: List[Thesis], add_topic_to_keywords: bool=True) -> "None":
        """
        Construct scorer using keywords for given theses.

        Parameters
        ----------
        theses: List[Thesis]
            theses to score against
        add_topic_to_keywords: bool
            whether to add tokenized topic to keywords
        """

        theses = clean_theses(theses)
        self._add_topic_to_keywords = add_topic_to_keywords
        self._theses_keywords = [
            self._extract_keywords(t)
            for t 
            in theses
            ]

    def score(self, thesis: Thesis) -> List[float]:
        
        thesis_keywords = self._extract_keywords(thesis)
        # calculate jaccard scores
        return [jaccard_distance(thesis_keywords, kw) for kw in self._theses_keywords]

    # extracts keywords
    def _extract_keywords(self, thesis: Thesis) -> Set[str]:

        return preprocess_keywords(
            thesis.topic.split(" ") + thesis.keywords 
            if self._add_topic_to_keywords 
            else thesis.keywords
            )


class ThesisScorerMetadata(ThesisScorerBase):
    """Thesis Scorer that uses cosine similarity for encoded metadata features."""

    def __init__(self, theses: List[Thesis], metadata_columns: List[str]=["reader", "supervisor"]) -> "None":
        """
        Construct scorer using metadata for given theses.

        Parameters
        ----------
        theses: List[Thesis]
            theses to score against
        metadata_columns: List[str]
            columns to use as metadata -- these have to be one hot encodable
        """
        
        self._metadata_columns = metadata_columns[:] # copy the default parameters
        # create encoders
        self._encoders = {col: OneHotEncoder([t.__dict__[col] for t in theses]) for col in metadata_columns}
        # encode
        self._metadata_encoded = [self._encode_metadata_features(t) for t in theses]

    def score(self, thesis: Thesis) -> List[float]:
        
        # encode
        thesis_metadata_features = self._encode_metadata_features(thesis)
        # calculate cosine similarity
        return [cosine_similarity(thesis_metadata_features, of) for of in self._metadata_encoded]

    def _encode_metadata_features(self, thesis: Thesis) -> List[bool]:
        
        encoded_result = []
        # encode and concat for each
        for col in self._metadata_columns:
            encoded_result.extend(self._encoders[col].encode(thesis.__dict__[col]))
        return encoded_result


class ThesisScorerBM25(ThesisScorerBase):
    """Thesis Scorer that uses text features for BM25 score."""

    def __init__(self, 
        theses: List[Thesis], 
        use_en_abstract: bool = True, 
        use_cs_abstract: bool = False, 
        use_description: bool = False, 
        use_topic: bool = False,
        k1: float=1.5,
        b: float=0.75, 
        delta: float=0.5
        ) -> "None":
        """
        Construct scorer using text data with BM25 for given theses.

        Parameters
        ----------
        theses: List[Thesis]
            theses to score against
        use_en_abstract: bool
            whether to use english abstract
        use_cs_abstract: bool
            whether to use czech abstract
        use_description: bool
            whether to use description
        use_topic: bool
            whether to use topic
        k1: float
            BM25+ parameter
        b: float
            BM25+ parameter
        delta: float
            BM25+ parameter
        """

        theses = clean_theses(theses)
        self._use_en_abstract = use_en_abstract
        self._en_abstract_stopwords = extract_docs_stopwords([t.en_abstract for t in theses], preprocess_en_abstract)
        self._use_cs_abstract = use_cs_abstract
        self._cs_abstract_stopwords = extract_docs_stopwords([t.cs_abstract for t in theses], preprocess_cs_abstract)
        self._use_description = use_description
        self._description_stopwords = extract_docs_stopwords([t.description for t in theses], preprocess_description)
        self._use_topic = use_topic
        self._topic_stopwords = extract_docs_stopwords([t.topic for t in theses], preprocess_topic)
        self._bm25 = BM25([self._extract_bm25_tokens(t) for t in theses], k1=k1, b=b, delta=delta)

    def score(self, thesis: Thesis) -> List[float]:
        
        # tokenize
        thesis_text_tokens = self._extract_bm25_tokens(thesis)
        # score with model
        return self._bm25.score(thesis_text_tokens)

    def _extract_bm25_tokens(self, thesis: Thesis) -> List[str]:

        cs_abstract = preprocess_cs_abstract(thesis.cs_abstract if self._use_cs_abstract else "", self._cs_abstract_stopwords, True)
        en_abstract = preprocess_en_abstract(thesis.en_abstract if self._use_en_abstract else "", self._en_abstract_stopwords, True)
        description = preprocess_description(thesis.description if self._use_description else "", self._description_stopwords, True)
        topic = preprocess_topic(thesis.topic if self._use_topic else "", self._topic_stopwords, True)
        return cs_abstract + en_abstract + description + topic


class ThesisScorerEnsemble(ThesisScorerBase):
    """Thesis Scorer that takes many scorers and ensembles them."""

    def __init__(self, scorers_with_weights: List[Tuple[ThesisScorerBase, float]], multiply_scores: bool=False) -> "None":
        """
        Constructs scorer that ensembles other scorers.

        Parameters
        ----------
        scorers_with_weights: List[Tuple[ThesisScorerBase, float]]
            list of tuples in format: scorer, weight; each scorer is weighted w given weight
        multiply_scores: bool
            whether to multiply scores, if False then they are summed up
        
        Note: all scorers should be initialized on the same theses! Otherwise the result is useless!
        """
        
        self._multiply_scores = multiply_scores
        self._scorers_with_weights = scorers_with_weights

    def score(self, thesis: Thesis) -> List[float]:
        
        # get score for each scorer
        scores_per_scorer = [scorer.score(thesis) for scorer, _ in self._scorers_with_weights]
        # convert the scores to inverted order scores (assign max points to first, second max to second ..)
        inverted_order_scores_per_scorer = [self._score_to_inverted_order_score(scores) for scores in scores_per_scorer]
        # merge the inverted order scores, either by sum or multiply (depends on constructor argument)
        weights = [w for _, w in self._scorers_with_weights]
        merged_inverted_order_scores = self._merge_scores(inverted_order_scores_per_scorer, weights)
        # sort them by index and throw away the index (we only want the scores)
        return [score for _, score in sorted(merged_inverted_order_scores, key=lambda item: item[0])]

    # partial score of one inner scorer non-weighted
    def score_partial(self, thesis: Thesis, scorer_index: int) -> List[float]:
        """Perform partial scoring by returning the scores of scorer under given index only.
        
        Parameters
        ----------
        thesis: Thesis
            thesis to score
        scorer_index: int
            index of scorer to use

        Returns
        -------
        List[float]
            list of scores
        """

        if scorer_index < 0 or scorer_index >= len(self._scorers_with_weights):
            return []
        return self._scorers_with_weights[scorer_index][0].score(thesis)

    # return (index, inverted order)
    def _score_to_inverted_order_score(self, scores: List[float]) -> List[Tuple[int, float]]:

        scores_with_indices = enumerate(scores)
        # sort with indices based on first score followed by index (so there is some ordering when the same score)
        # so this is (index, score)
        sorted_scores_with_indices = sorted(scores_with_indices, key=lambda item: (item[1], item[0]), reverse=True)
        # remove old scores, indices are sorted based on the score here
        sorted_indices = (i for i, _ in sorted_scores_with_indices)
        # add order scores and invert them (the first one gets the most points!)
        max_inverted_order_score = len(scores)
        return [(i, max_inverted_order_score - order) for order, i in enumerate(sorted_indices)]
        
    # merge scores of scorers using given weights
    def _merge_scores(self, 
        scores_per_scorer: List[List[Tuple[int, float]]], 
        weights_per_scorer: List[float]
        ) -> List[Tuple[int, float]]:

        if not scores_per_scorer:
            return []

        length = max([len(scores) for scores in scores_per_scorer])
        # initial scores, once for multiplication, zeros for sum!
        if self._multiply_scores:
            result_scores = [1] * length
        else:
            result_scores = [0] * length

        for scores, weight in zip(scores_per_scorer, weights_per_scorer):
            if weight <= 0:
                continue
            for i, score in scores:
                # invert order so bigger is better
                weighted_score = score * weight
                if self._multiply_scores:
                    result_scores[i] *= weighted_score
                else: 
                    result_scores[i] += weighted_score
        # add indices once again and cast to list
        return [*enumerate(result_scores)]


# RECOMMENDER SYSTEM CLASSES

class ThesisRecommenderSystem(ThesisRecommenderSystemBase):
    """Thesis Recommender System that takes a scorer and recommends based on its scores."""

    def __init__(self, theses: List[Thesis], scorer: ThesisScorerBase) -> "None":
        """
        Creates recommender system for the given `theses`.

        Parameters
        ----------
        theses: List[Thesis]
            List of all theses that can be used as possible recommendations.
        scorer: ThesisScorerBase
            Valid ThesisScorerBase object that is used to score the theses.
        """

        self._index2thesis = dict(enumerate(theses))
        self._scorer = scorer
    
    def recommend(self, thesis: Thesis, n_recommendations: int=10) -> List[Tuple[int, Thesis]]:

        # get scores
        scores = self._scorer.score(thesis)
        # add indices
        scores_with_indices = enumerate(scores)
        # sort with indices based on first score followed by index (so there is some ordering when the same score)
        sorted_scores_with_indices = sorted(scores_with_indices, key=lambda item: (item[1], item[0]), reverse=True)

        recommendations = []
        for i, _ in sorted_scores_with_indices:
                
            recommend_candidate = self._index2thesis[i]
            # its important we don't recommend the exact same thesis and this check should do that
            if thesis != recommend_candidate:
                recommendations.append((i, recommend_candidate))
            if len(recommendations) == n_recommendations:
                break
        
        return recommendations 


class ThesisRecommenderSystemKeywords(ThesisRecommenderSystem):
    """Thesis recommender system that uses keywords."""
    
    def __init__(self, theses: List[Thesis], add_topic_to_keywords: bool = True) -> "None":
        """
        Construct recommender using keywords for given theses.

        Parameters
        ----------
        theses: List[Thesis]
            theses to score against
        add_topic_to_keywords: bool
            whether to add tokenized topic to keywords
        """

        super(ThesisRecommenderSystemKeywords, self).__init__(
            theses, 
            ThesisScorerKeywords(theses, add_topic_to_keywords=add_topic_to_keywords)
            )


class ThesisRecommenderSystemMetadata(ThesisRecommenderSystem):
    """Thesis recommender system that uses encoded metadata."""
    
    def __init__(self, theses: List[Thesis], metadata_columns: List[str]=["reader", "supervisor"]) -> "None":
        """
        Construct recommender using metadata for given theses.

        Parameters
        ----------
        theses: List[Thesis]
            theses to score against
        metadata_columns: List[str]
            columns to use as metadata -- these have to be one hot encodable
        """

        super(ThesisRecommenderSystemMetadata, self).__init__(
            theses, 
            ThesisScorerMetadata(theses, metadata_columns=metadata_columns)
            )


class ThesisRecommenderSystemBM25(ThesisRecommenderSystem):
    """Thesis recommender system that uses BM25 text data."""
    
    def __init__(self, 
        theses: List[Thesis], 
        use_en_abstract: bool = True, 
        use_cs_abstract: bool = False, 
        use_description: bool = False, 
        use_topic: bool = False,
        k1: float=1.5,
        b: float=0.75, 
        delta: float=0.5
        ) -> "None":
        """
        Construct thesis recommender using text data with BM25 for given theses.

        Parameters
        ----------
        theses: List[Thesis]
            theses to score against
        use_en_abstract: bool
            whether to use english abstract
        use_cs_abstract: bool
            whether to use czech abstract
        use_description: bool
            whether to use description
        use_topic: bool
            whether to use topic
        k1: float
            BM25+ parameter
        b: float
            BM25+ parameter
        delta: float
            BM25+ parameter
        """

        super(ThesisRecommenderSystemBM25, self).__init__(
            theses, 
            ThesisScorerBM25(
                theses,
                use_en_abstract=use_en_abstract,
                use_cs_abstract=use_cs_abstract,
                use_description=use_description,
                use_topic=use_topic,
                k1=k1,
                b=b,
                delta=delta
                )
            )


class ThesisRecommenderSystemEnsemble(ThesisRecommenderSystem):
    """Thesis recommender system that relies on ensembling of scorers."""

    def __init__(self, 
        theses: List[Thesis], 
        scorers_with_weights: List[Tuple[ThesisScorerBase, float]],
        multiply_scores: bool=False # sum them up
        ) -> "None":
        """
        Constructs recommender that relies on ensembling scorers.

        Parameters
        ----------
        scorers_with_weights: List[Tuple[ThesisScorerBase, float]]
            list of tuples in format: scorer, weight; each scorer is weighted w given weight
        multiply_scores: bool
            whether to multiply scores, if False then they are summed up
        
        Note: all scorers should be initialized on the same theses! Otherwise the result is useless!
        """

        super(ThesisRecommenderSystemEnsemble, self).__init__(
            theses, 
            ThesisScorerEnsemble(scorers_with_weights, multiply_scores=multiply_scores)
            )


# UTILS

def normalize_scores(scores: List[float]) -> List[float]:
    """Normalize scores dividing by the max."""
    
    max_score = max(scores)
    return [score / max_score for score in scores]

def normalize_lists_of_scores(scores_list: List[List[float]]) -> List[float]:
    """Normalize list of scores dividing by the global maximum."""
    
    max_score = max([max(scores) for scores in scores_list])
    return [[score / max_score for score in scores] for scores in scores_list]

