from typing import List


class OneHotEncoder:
    """
    Class that is used to one hot encode categories

    Attributes
    -----------
    sorted_categories: List[str]
        sorted categories that are used by encoder
    """

    def __init__(self, categories: List[str]) -> "None":
        """
        Initializes encoder for given categories

        Note: categories are sorted

        Parameters
        -----------
        categories: List[str]
            list of categories to encode against
        """
        
        self.sorted_categories = sorted(list(set(categories)))
    
    def encode(self, category: str) -> List[bool]:
        """
        Encodes category into a list of 0s or 1s depending whether it fits supplied categories in constructor

        Parameters
        -----------
        category: str
            category string to encode
        
        Returns
        -----------
        List[bool]
            boolean mask, each 1 means that category fits sorted_categories on given index
        """

        return [
            category == indexed_category
            for indexed_category
            in self.sorted_categories
            ]
