from pathlib import Path
from typing import Union, List, Dict

from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from clustering.rate_beer_loader import RateBeerLoader

class RateBeerPykeen:
    """
    This class will use Pykeen to create a model in the same manner as the
    Unsupervised Customer Clustering paper has.
    What I need to do:
        1. File load
        2. Turn into triple set
        3. Create a pipeline
        4. Use the data to group the customers
    """

    def __init__(self, file_location: Union[str, Path]):
        self._file_location = Path(file_location)
        self._rate_beer_loader = RateBeerLoader(self._file_location)

        # The training loop 'sLCWA' is to use negative sampling for training.
        self.pipeline = pipeline(dataset=self._rate_beer_loader.get_rate_beer_pykeen_format(),
                                 model="TransE",
                                 training_loop='sLCWA',
                                 negative_sampler='basic')

    def fit(self):
        """

        """
        pass
