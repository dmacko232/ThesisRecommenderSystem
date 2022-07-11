# Cleaning functions
from collections import defaultdict
from collections.abc import Collection
from os import remove
from load_data import Thesis
from functools import partial

from typing import List, Set, Union, Callable
from gensim.parsing.preprocessing import (
    preprocess_string, 
    strip_tags, 
    strip_punctuation, 
    strip_multiple_whitespaces, 
    strip_numeric, 
    remove_stopwords, 
    stem_text
    )

def clean_thesis(thesis: Thesis) -> Thesis:

    # strip topic
    thesis.topic = thesis.topic.strip()
    # strip cs_abstract
    thesis.cs_abstract = thesis.cs_abstract.strip()
    # strip en_abstract
    thesis.en_abstract = thesis.cs_abstract.strip()
    # strip reader
    thesis.reader = thesis.reader.strip() 
    # strip description
    #thesis.supervisor = thesis.supervisor.strip() 
    # strip description
    #thesis.description = thesis.description.strip()
    return thesis 

def clean_theses(theses: Collection[Thesis]) -> Collection[Thesis]:

    return [clean_thesis(thesis) for thesis in theses]

def preprocess_keywords(keywords: List[str]) -> Set[str]:

    return set([
        kw.strip()
        for kw in keywords
        ])

def preprocess_cs_abstract(doc: str, stopwords: Set[str], stem: bool) -> List[str]:

    stopwords_func = partial(remove_stopwords, stopwords=stopwords)
    filters = [lambda s: s.lower(), strip_tags, strip_punctuation, 
               strip_multiple_whitespaces, strip_numeric, stopwords_func]
    if stem:
        filters.append(stem_text)
    return preprocess_string(doc, filters)

def preprocess_en_abstract(doc: str, stopwords: Set[str], stem: bool) -> List[str]:

    stopwords_func = partial(remove_stopwords, stopwords=stopwords)
    filters = [lambda s: s.lower(), strip_tags, strip_punctuation, 
               strip_multiple_whitespaces, strip_numeric, stopwords_func]
    if stem:
        filters.append(stem_text)
    return preprocess_string(doc, filters)

def preprocess_description(doc: str, stopwords: Set[str], stem: bool) -> List[str]:

    stopwords_func = partial(remove_stopwords, stopwords=stopwords)
    filters = [lambda s: s.lower(), strip_tags, strip_punctuation, 
               strip_multiple_whitespaces, strip_numeric, stopwords_func]
    if stem:
        filters.append(stem_text)
    return preprocess_string(doc, filters)

def preprocess_topic(doc: str, stopwords: Set[str], stem: bool) -> List[str]:

    stopwords_func = partial(remove_stopwords, stopwords=stopwords)
    filters = [lambda s: s.lower(), strip_tags, strip_punctuation, 
               strip_multiple_whitespaces, strip_numeric, stopwords_func]
    if stem:
        filters.append(stem_text)
    return preprocess_string(doc, filters)

def extract_docs_stopwords(
    docs: List[str],
    preprocess_function: Callable[[str, Set[str], bool], List[str]],
    min_doc_freq: Union[int, float] = 5, 
    max_doc_freq: Union[int, float] = -5,
    ) -> Set[str]:

    length = len(docs)
    if isinstance(min_doc_freq, float):
        min_thresh = length * min_doc_freq
    else:
        min_thresh = min_doc_freq
    
    if isinstance(max_doc_freq, float):
        max_thresh = length * max_doc_freq
    elif max_doc_freq >= 0:
        max_thresh = max_doc_freq
    else: # max_doc_freq is -5 so we take length + (-5) = length - 5
        max_thresh = length + max_doc_freq

    doc_freqs = defaultdict(lambda: 0)
    for doc in docs:
        for token in set(preprocess_function(doc, set(), False)):
            doc_freqs[token] += 1
    
    return set([tok for tok, freq in doc_freqs.items() if freq < min_thresh or freq > max_thresh])
    