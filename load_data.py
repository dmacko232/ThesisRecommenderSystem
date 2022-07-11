import pickle
from collections.abc import Collection
from csv import DictReader
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, ClassVar
import builtins

# https://stackoverflow.com/questions/56797480/python-get-type-object-from-its-name
def get_type(type_name: str) -> type: 
    """Get type object from the name of tpe supplied"""
    try: 
        return getattr(builtins, str(type_name)) 
    except AttributeError: 
        try: 
            obj = globals()[type_name] 
        except KeyError:
            return None 
        return obj


class Language(Enum):
    """Language enumerable."""
    cze = auto()
    sla = auto()
    eng = auto()


class Evaluation(Enum):
    """Evaluation enumerable."""
    A = auto()
    B = auto()
    C = auto()
    D = auto()
    E = auto()
    F = auto()
    O = auto()  # Obhájeno/Defended Rigo
    P = auto()  # Prospěl/Passed Rigo


@dataclass
class Thesis:
    author_name: str        # Jméno
    author_id: int          # UČO
    topic: str              # Téma ZP
    language: Language      # Jazyk práce
    cs_abstract: str        # Anotace česky
    en_abstract: str        # Anotace anglicky
    keywords: list[str]     # Klíčová slova
    evaluation: Evaluation  # Hodnocení
    reader: str             # Oponent
    supervisor: str         # Vedoucí
    description: str        # Zadání práce

    field_dict: ClassVar[dict[str, str]] = {
        "Jméno": "author_name",
        "UČO": "author_id",
        "Téma ZP": "topic",
        "Jazyk práce": "language",
        "Anotace česky": "cs_abstract",
        "Anotace anglicky": "en_abstract",
        "Klíčová slova": "keywords",
        "Hodnocení": "evaluation",
        "Oponent": "reader",
        "Vedoucí": "supervisor",
        "Zadání práce": "description",
    }

    @classmethod
    def from_csv_dict(cls, csv_dict: dict[str, str]) -> "Thesis":  # no Self type till Python 3.11 :(
        """Reads from csv dictionary."""

        d: dict[str, Any] = {cls.field_dict[k]: v for k, v in csv_dict.items()}
        d["author_id"] = int(d["author_id"])
        d["language"] = Language[d["language"]]
        d["keywords"] = d["keywords"].split(", ")
        d["evaluation"] = Evaluation[d["evaluation"]]
        return cls(**d)

    def type_check(self) -> None:
        """Performs type check."""

        for field, value in self.__dict__.items():
            if field == "keywords":
                for word in value:
                    if not isinstance(word, str):
                        raise RuntimeError(f"{type(value)}, {field}, {value}" )
                break
            elif not isinstance(value, get_type(self.__class__.__annotations__[field])):
                raise RuntimeError(f"{type(value)}, {field}, {value}" )


def parse_theses(filename: str="sample_data.csv", used_cached: bool=True) -> Collection[Thesis]:
    """Parse theses into collection."""

    cache_filename = f"{filename}.p"
    if used_cached:
        try:
            with open(cache_filename, "rb") as cache_file:
                pickle.load(cache_file)
        except FileNotFoundError:
            pass
    out = []
    with open(filename, "r") as input_file:
        for line in DictReader(input_file):
            t = Thesis.from_csv_dict(line)
            #t.type_check()
            out.append(t)
    with open(cache_filename, "wb") as cache_file:
        pickle.dump(out, cache_file)
    return out
