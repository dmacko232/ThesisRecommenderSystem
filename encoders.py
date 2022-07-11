from typing import List


class OneHotEncoder():

    def __init__(self, categories: List[str]) -> "None":
        
        self._sorted_categories = sorted(categories)
    
    def encode(self, category: str) -> List[bool]:

        return [
            category == indexed_category
            for indexed_category
            in self._sorted_categories
            ]
