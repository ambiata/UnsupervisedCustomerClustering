"""
Loads in files
"""
from datetime import datetime
from pathlib import Path
from typing import Dict, Union, List

import numpy as np
from pykeen.datasets import Dataset
from pykeen.triples import TriplesFactory

_relationship_to_id_mapper = {
    "precedes": 0,
    "succeeds": 1,
    "appearance": 2,
    "aroma": 3,
    "palate": 4,
    "taste": 5,
    "overall": 6,
    "profileName": 7,
    "name": 8,
    "beerId": 9,
    "brewerId": 10,
    "ABV": 11,
    "style": 12
}


class RateBeerLoader:
    def __init__(self, file_location: Union[Path, str]):
        self._file_location = Path(file_location)
        self._rate_beer_processed = self._load_rate_beer()
        self._entity_id_mapper = None

    def __len__(self):
        return len(self._rate_beer_processed)

    def get_rate_beer(self):
        return self._rate_beer_processed

    def _load_rate_beer(self) -> List[Dict[str, Union[Dict[str, str]]]]:
        raw_rate_beer_dict = self._load_rate_beer_raw()
        rate_beer = _create_date_details(raw_rate_beer_dict)
        rate_beer = _create_review_id(rate_beer)
        rate_beer = _connect_reviews_using_id(rate_beer)
        return rate_beer

    def _load_rate_beer_raw(self) -> List[Dict[str, Dict[str, str]]]:
        """
        Loads in the rate beer file and fields.
        Returns:
            The reviews as a list of dictionaries of reviews, everything is just in string format.
            Each review is a dictionary of dictionaries in the form:
            {"beer": {name: value}, "review": {name: value}}
        """
        path_to_file = Path(self._file_location)
        reviews = []
        current_rating = {"review": {}, "beer": {}}
        with path_to_file.open(mode="r") as rate_beer_file:
            # Reads in all the lines until an empty line, then combine those into a dictionary.
            for line_raw in rate_beer_file.readlines():
                line = line_raw.rstrip().lstrip()
                if line == "" and len(current_rating["review"]) > 0:
                    reviews.append(current_rating)
                    current_rating = {"review": {}, "beer": {}}
                elif line.startswith("beer"):
                    key, value = _get_line_key_value(line, "beer")
                    current_rating["beer"][key] = value
                else:
                    key, value = _get_line_key_value(line, "review")

                    # We ignore the text reviews for this paper.
                    if key == "text":
                        continue
                    current_rating["review"][key] = value
            if len(current_rating["review"]) > 0:
                reviews.append(current_rating)
        return reviews

    def get_rate_beer_pykeen_format(self, clear_map=False) -> Dataset:
        """
        Pykeen requires an n x 3 numpy array. With the form (Head id, Relationship id, Tail id)
        where the ids are a representation of the relationships and entities.

        Returns:
            n x 3 integer array.
        """
        # Create a mapping if there is no mapper
        if self._entity_id_mapper is None or clear_map:
            self._create_pykeen_entity_id_mapper()

        head_relationship_tail_representation = self._create_pykeen_hrt_representation()
        hrt_triples_factory = TriplesFactory(head_relationship_tail_representation,
                                             entity_to_id=self._entity_id_mapper,
                                             relation_to_id=_relationship_to_id_mapper)
        Dataset.from_tf(hrt_triples_factory, [1.0, 0.0, 0.0])
        return head_relationship_tail_representation

    def _create_pykeen_entity_id_mapper(self):
        """
        This will process the relationships and entities to integer ids and record a mapper.
        This is for use with the Pykeen library.
        """
        # Create a set of all the entities.
        entities_seen = set()
        for review in self._rate_beer_processed:
            # Take each of the node values and turn them into an integer value.
            review_values = [f"{key[:2]}{value}" for key, value in review["review"].items()]
            beer_values = [f"{key[:2]}{value}" for key, value in review["beer"].items()]

            entities_seen.union(review_values)
            entities_seen.union(beer_values)

        # Convert the entity set to an entity map.
        self._entity_id_mapper = {entity_str: integer_value
                                  for integer_value, entity_str in enumerate(list(entities_seen))}

    def _create_pykeen_hrt_representation(self) -> np.ndarray:
        beer_rate = self._rate_beer_processed.copy()
        head_rel_tail = []

        for review in beer_rate:
            review_id = review["id"]
            head_rel_tail = _add_triples("review", head_rel_tail, review_id, review)
            head_rel_tail = _add_triples("beer", head_rel_tail, review_id, review)

            if "precedes" in review.keys():
                triple = [review_id, _relationship_to_id_mapper["precedes"], review["precedes"]]
                head_rel_tail.append(triple)

            if "succeeds" in review.keys():
                triple = [review_id, _relationship_to_id_mapper["succeeds"], review["succeeds"]]
                head_rel_tail.append(triple)

        head_rel_tail = np.array(head_rel_tail)
        return head_rel_tail


def _get_line_key_value(line, category):
    stripped = line.replace(f"{category}/", "")
    # The format will now be "key: value" so split based on this.
    key_value_pair = stripped.split(":")
    key = key_value_pair[0]
    # There could have been ':'s in the value.
    value = ":".join(key_value_pair[1:]).lstrip()
    return key, value


def _create_date_details(rate_beer_review_list):
    """
    Creates the fields of Year, Month and DayOfWeek for each review.

    Args:
        rate_beer_review_list: A list of the reviews.

    Returns:
        The list of reviews with information extracted from the time field's value.
    """
    rate_beer = rate_beer_review_list.copy()
    for i in range(len(rate_beer_review_list)):
        time_value_as_int = int(rate_beer_review_list[i]["review"]["time"])
        date_timestamp = datetime.fromtimestamp(time_value_as_int)
        weekday = date_timestamp.weekday()
        year = date_timestamp.year
        month = date_timestamp.month
        rate_beer[i]["review"]["Year"] = str(year)
        rate_beer[i]["review"]["DayOfWeek"] = str(weekday)
        rate_beer[i]["review"]["Month"] = str(month)
    return rate_beer


def _create_review_id(rate_beer):
    """
    The review needs to have an id for us to link each of these reviews. This
    will be created by sorting the reviews by ascending time order.

    Args:
        rate_beer: The list of ratings.

    Returns:
        The list of reviews with an 'id' key-value of the reviews.
    """
    sorted_reviews = sorted(rate_beer, key=lambda review: int(review["review"]["time"]))
    for i in range(len(sorted_reviews)):
        sorted_reviews[i]["id"] = str(i)
        del sorted_reviews[i]["review"]["time"]
    return sorted_reviews


def _connect_reviews_using_id(rate_beer):
    """
    The reviews are connected forwards and backwards to each other which this will provide
    Args:
        rate_beer: A sorted list of the reviews

    Returns:
        A sorted list of the reviews with a "succeeds" and "precedes" value.
    """
    rate_beer = rate_beer.copy()
    for i in range(len(rate_beer)):
        if i != 0:
            rate_beer[i]["succeeds"] = str(rate_beer[i - 1]["id"])
        if i != len(rate_beer) - 1:
            rate_beer[i]["precedes"] = str(rate_beer[i + 1]["id"])
    return rate_beer


def _add_triples(key, head_rel_tail, review_id, review):
    for field, value in review[key].items():
        head_rel_tail.append([review_id, _relationship_to_id_mapper[field], value])
    return head_rel_tail
