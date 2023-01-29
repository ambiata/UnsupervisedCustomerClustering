"""
Test the loaders abilities
"""
from pathlib import Path

import pytest
import unittest
import clustering.rate_beer_loader as rb_loader

rate_beer_first_two = \
    """beer/name: John Harvards Simcoe IPA
beer/beerId: 63836
beer/brewerId: 8481
beer/ABV: 5.4
beer/style: India Pale Ale &#40;IPA&#41;
review/appearance: 4/5
review/aroma: 6/10
review/palate: 3/5
review/taste: 6/10
review/overall: 13/20
review/time: 1157587200
review/profileName: hopdog
review/text: On tap at the Springfield, PA location.

beer/name: John Harvards Simcoe IPA
beer/beerId: 63836
beer/brewerId: 8481
beer/ABV: 5.4
beer/style: India Pale Ale &#40;IPA&#41;
review/appearance: 4/5
review/aroma: 6/10
review/palate: 4/5
review/taste: 7/10
review/overall: 13/20
review/time: 1157241600
review/profileName: TomDecapolis
review/text: On tap at the John Harvards in Springfield PA.

beer/name: John Harvards Cristal Pilsner
beer/beerId: 71716
beer/brewerId: 8481
beer/ABV: 5
beer/style: Bohemian Pilsener
review/appearance: 4/5
review/aroma: 5/10
review/palate: 3/5
review/taste: 6/10
review/overall: 14/20
review/time: 958694400
review/profileName: PhillyBeer2112
review/text: UPDATED: FEB 19, 2003 Springfield, PA.

beer/name: John Harvards Fancy Lawnmower Beer
beer/beerId: 64125
beer/brewerId: 8481
beer/ABV: 5.4
beer/style: K�lsch
review/appearance: 2/5
review/aroma: 4/10
review/palate: 2/5
review/taste: 4/10
review/overall: 8/20
review/time: 1157587200
review/profileName: TomDecapolis
review/text: On tap the Springfield PA location billed as the "Fancy Lawnmower Light".
"""

expected_beer_results = [
    {
        "review": {"appearance": "4/5",
                   "aroma": "5/10",
                   "palate": "3/5",
                   "taste": "6/10",
                   "overall": "14/20",
                   "profileName": "PhillyBeer2112",
                   "Year": "2000",
                   "DayOfWeek": "4",
                   "Month": "5"
                   },
        "beer": {
            "beerId": "71716",
            "style": "Bohemian Pilsener"},
        "id": "0",
        "precedes": "1",
    },
    {
        "review": {
            "appearance": "4/5",
            "aroma": "6/10",
            "palate": "4/5",
            "taste": "7/10",
            "overall": "13/20",
            "profileName": "TomDecapolis",
            "Year": "2006",
            "DayOfWeek": "6",
            "Month": "9"
        },
        "beer": {
            "beerId": "63836",
            "style": "India Pale Ale &#40;IPA&#41;"},
        "id": "1",
        "precedes": "2",
        "succeeds": "0"
    },
    {
        "review": {
            "appearance": "4/5",
            "aroma": "6/10",
            "palate": "3/5",
            "taste": "6/10",
            "overall": "13/20",
            "profileName": "hopdog",
            "Year": "2006",
            "DayOfWeek": "3",
            "Month": "9"
        },
        "beer": {
            "beerId": "63836",
            "style": "India Pale Ale &#40;IPA&#41;"
        },
        "id": "2",
        "succeeds": "1",
        "precedes": "3",
    },
    {
        "review": {"appearance": "2/5",
                   "aroma": "4/10",
                   "palate": "2/5",
                   "taste": "4/10",
                   "overall": "8/20",
                   "profileName": "TomDecapolis",
                   "Year": "2006",
                   "DayOfWeek": "3",
                   "Month": "9"},
        "beer": {"beerId": "64125",
                 "style": "K�lsch"},
        "id": "3",
        "succeeds": "2",
    },

]

location_for_file = Path("rate_beer_torch.txt").absolute()


@pytest.fixture(autouse=True)
def create_temp_beer_file():
    with location_for_file.open("w") as temp_beer_file:
        temp_beer_file.write(rate_beer_first_two)
    yield
    location_for_file.unlink()


def test_load_beer_file():
    """
    When the beer file is loaded I expect three reviews with the above ratings.
    """
    reviews = rb_loader.RateBeerLoader(location_for_file)
    number_detected_reviews = len(reviews)
    assert number_detected_reviews == len(expected_beer_results), \
        f"The number of reviews returned {number_detected_reviews}, is not equal to the number " \
        f"expected {len(expected_beer_results)}"
    for i in range(number_detected_reviews):
        expected_result = expected_beer_results[i]
        review_value = reviews.get_rate_beer_raw()[i]
        unittest.TestCase().assertDictEqual(review_value, expected_result)


