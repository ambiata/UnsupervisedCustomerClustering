from pathlib import Path

from clustering.rate_beer_loader import RateBeerLoaderPykeen


def test_get_rate_beer_plain():
    """
    This will test the Pykeen rate beer loader to ensure it can load the data
    in the format needed.
    """
    beer_location = Path(__file__).parent.joinpath("rate_beer_torch.txt").absolute()
    beer_location = Path(__file__).parents[1].joinpath("ratebeer.txt").absolute()
    loader = RateBeerLoaderPykeen(beer_location)
    rate_beer_res = loader.get_rate_beer()
