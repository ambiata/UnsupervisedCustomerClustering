"""
For creating and loading triples of the files.
"""
from pathlib import Path
from typing import Union
from clustering.rate_beer_loader import load_rate_beer_raw

class RateBeerTripleCreator:
    """
    For creating the triples of the graph listed in the paper
    Unsupervised Customer Segmentation with Knowledge Graph Embeddings on page 2 Figure 1-b
    """

    def __init__(self, file_location: Union[str, Path]):
        """
        Converts a file from RateBeer into triples of form (entity1, relationship, entity2)

        Args:
            file_location: The location of ratebeer.txt file.
        """
        self.file_location = Path(file_location)
        self._file_loaded = load_rate_beer_raw(self.file_location)
