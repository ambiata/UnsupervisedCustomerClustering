"""
Loads in files
"""
from random import choices
from datetime import datetime
from pathlib import Path
from typing import Dict, Union, List, Tuple
from multiprocessing import Pool

import numpy as np
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
    "beerId": 8,
    "style": 9,
    "Year": 10,
    "DayOfWeek": 11,
    "Month": 12
}


def _remove_beer_specific_details(rate_beer):
    """
    The beer reviews contain the fields "ABV"
    "name", and "brewerId" were not included in the
    graph used by the paper, and they are removed here.

    Args:
        rate_beer: The reviews in a list of dictionary format.

    Returns:
        Rate Beer dataset with the fields "ABV", "name" and "brewerId"
        removed.
    """

    for review in rate_beer:
        del review["beer"]["ABV"]
        del review["beer"]["name"]
        del review["beer"]["brewerId"]
    return rate_beer


class RateBeerLoader:
    def __init__(self, file_location: Union[Path, str]):
        self.all_reviewers = set()
        self._file_location = Path(file_location)
        self._rate_beer_processed = []

    def __len__(self):
        return len(self._rate_beer_processed)

    def get_rate_beer_raw(self):
        return self._rate_beer_processed

    def load_rate_beer(self, ) -> List[Dict[str, Union[Dict[str, str]]]]:
        """
        Loads and transforms raw data into a processable form following the article framework,
        with the following steps:

        1. Loads the raw data
        2. Transforms the time field into `Year`, `Month`, `DayOfWeek`.
        3. Sorts the reviews by time giving each review an id based on this.
        4. Deletes the time fields as it is now in the prior information.
        5. Connects each of the reviews by a precedes or a succeeds relationship.

        Returns:
            The processed rate_beer.
        """
        raw_rate_beer_dict = self._load_rate_beer_raw()
        rate_beer = _create_date_details(raw_rate_beer_dict)
        rate_beer = _create_review_id(rate_beer)
        rate_beer = self._connect_reviews_using_id(rate_beer)
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
        with path_to_file.open(mode="r",
                               encoding="utf-8",
                               errors="replace") as rate_beer_file:
            # Reads in all the lines until an empty line, then combine those into a dictionary.
            for line_raw in rate_beer_file.readlines():
                line = line_raw.rstrip().lstrip()
                if line == "" and len(current_rating["review"]) > 0:
                    # Append the current rating and append a new one.
                    reviews.append(current_rating)
                    current_rating = {"review": {}, "beer": {}}
                elif line.startswith("beer"):
                    key, value = _get_line_key_value(line, "beer")
                    if key in ["ABV", "name", "brewerId"]:
                        continue
                    current_rating["beer"][key] = value
                else:
                    key, value = _get_line_key_value(line, "review")

                    # We ignore the text reviews for this paper.
                    if key == "text":
                        continue
                    if key == "profileName":
                        self.all_reviewers.add(value)

                    current_rating["review"][key] = value
            if len(current_rating["review"]) > 0:
                reviews.append(current_rating)
        return reviews

    def _connect_reviews_using_id(self, rate_beer):
        """
        The reviews of each reviewer are connected forwards and backwards to each other.
        This approach connects the reviews in time using graph abilities.

        Args:
            rate_beer: A sorted list of the reviews

        Returns:
            A sorted list of the reviews with a "succeeds" and "precedes" value.
        """
        rate_beer = rate_beer.copy()
        reviewers_most_recent_review = {reviewer: [] for reviewer in self.all_reviewers}

        # Obtain a dict of the index of all the reviews performed by each reviewer (consumer)
        for i in range(len(rate_beer)):
            observing_reviewer = rate_beer[i]["review"]["profileName"]
            reviewers_most_recent_review[observing_reviewer].append(i)

        # Take the lists of all the reviews for each reviewer and bond them together.
        for review_indices in reviewers_most_recent_review.values():
            # If they only have a single review the review doesn't proceed or succeed another review.
            if len(review_indices) == 1:
                continue
            largest_id_value = review_indices[-1]
            smallest_id_value = review_indices[0]
            for i in range(len(review_indices)):

                index_of_review = review_indices[i]
                if index_of_review != smallest_id_value:
                    # There is a review this person submitted before this one.
                    index_of_previous_review = review_indices[i - 1]
                    rate_beer[index_of_review]["succeeds"] = str(rate_beer[index_of_previous_review]["id"])
                if index_of_review != largest_id_value:
                    index_of_next_review = review_indices[i + 1]
                    rate_beer[index_of_review]["precedes"] = str(rate_beer[index_of_next_review]["id"])
        return rate_beer


def write_to_tsv(head_relationship_tail_representation: List[Tuple[str, str, str]]) -> Path:
    """
    Writes each <head relationship tail> triple to a tsv file ready for PyKeen\

    Args:
        head_relationship_tail_representation: A list of the head, relationship, tail tuples.

    Returns:
        The location the file was written to.
    """
    tab_seperated = ["\t".join(val) for val in head_relationship_tail_representation]
    tsv_file = Path("TEMPORARY_HRT.tsv").absolute()
    with tsv_file.open("w") as values:
        for line in tab_seperated:
            values.write(line + "\n")
    return tsv_file


class RateBeerLoaderPykeen(RateBeerLoader):
    """
    Deals with pytorch and pykeen specific setup of the dataset.
    The data is loaded and the information is written to a TSV file for the TriplesFactory to
    use.

    Imports are kept local to ensure that not having Pykeen or Pytorch installed
    doesn't cause issues if you attempt to try Tensorflow approach.
    """

    def __init__(self, file_location: Union[Path, str], checkpoint_name: str, accept_previous_saves: bool = True):
        super().__init__(file_location)

        self.checkpoint_name = checkpoint_name
        self._temporary_training_location = Path("training_file.tsv").absolute()
        self._head_relationship_tail_representation: List[Tuple[str, str, str]] = []
        if accept_previous_saves and self._temporary_training_location.exists():
            print("Found a previous save")
        else:
            print("Beginning a new read.")
            self._rate_beer_processed = self.load_rate_beer()
            self._head_relationship_tail_representation = self._create_hrt_list()
            self._write_temporary_training_file()

        print("Loading Triples from the path downloaded.")
        self._training_triples_factory = self._load_training_factory()

    def _load_training_factory(self):
        from pykeen.constants import PYKEEN_CHECKPOINTS
        import torch

        previous_checkpoint = PYKEEN_CHECKPOINTS.joinpath(self.checkpoint_name)
        entity_to_id = None
        relationship_to_id = None
        if previous_checkpoint.exists():
            loaded = torch.load(previous_checkpoint)
            entity_to_id = loaded["entity_to_id_dict"]
            relationship_to_id = loaded["relation_to_id_dict"]
        return TriplesFactory.from_path(self._temporary_training_location,
                                        entity_to_id=entity_to_id,
                                        relation_to_id=relationship_to_id)

    def _write_temporary_training_file(self):
        """
        Writes the <h,r,t> as a tab separated file to use.
        """
        with self._temporary_training_location.open("w") as training_file:
            for line in self._head_relationship_tail_representation:
                tsv_line = "\t".join(line)
                training_file.write(tsv_line + "\n")
        print("Written Temporary File")

    def get_rate_beer(self) -> Union[tuple[TriplesFactory, TriplesFactory, TriplesFactory],
                                     TriplesFactory]:
        """
        Gets the triples factory of the training set.

        Returns:
            The training and the first testing triples factory.
        Examples:
            loader = RateBeerLoaderPykeen("mybeer.txt")
            tf_training, tf_testing, tf_eval = loader.get_rate_beer()

            pipeline_result_complex = pipeline(training=tf_training, testing=tf_testing, model="ComplEx",
                                   training_loop="LCWA", model_kwargs=dict(embedding_dim=50))
            pipeline_result_trans_e = pipeline(training=tf_training, testing=tf_testing, model="TransE",
                                   training_loop="LCWA", model_kwargs=dict(embedding_dim=50))
        """
        return self._training_triples_factory

    def _create_hrt_list(self) -> List[Tuple[str, str, str]]:
        """
        Creates a list that is ready to be written in the form:
        <head, relationship, tail>

        Returns:
            A list of the graph nodes.
        """
        head_rel_tail = []

        assert self._rate_beer_processed, "The graph list is empty."
        for review in self._rate_beer_processed:
            review_id = review["id"]
            head_rel_tail = _add_triples("review", head_rel_tail, review_id, review)
            head_rel_tail = _add_triples("beer", head_rel_tail, review_id, review)

            if "precedes" in review.keys():
                triple = [review_id, "pre", review["precedes"]]
                head_rel_tail.append(triple)
            if "succeeds" in review.keys():
                triple = [review_id, "suc", review["succeeds"]]
                head_rel_tail.append(triple)
        return head_rel_tail


class RateBeerLoaderLSTM(RateBeerLoader):
    def __init__(self, file_location: Union[Path, str]):
        """
        This requires load in the data turn the graph into sequences for each person and then load them.
        """
        super().__init__(file_location)


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
        rate_beer[i]["review"]["Year"] = "yr_" + str(year)
        rate_beer[i]["review"]["DayOfWeek"] = "wk_" + str(weekday)
        rate_beer[i]["review"]["Month"] = "mon_" + str(month)
    return rate_beer


def _create_review_id(rate_beer):
    """
    The review needs to have an id for us to link each of these reviews. This
    will be created by sorting the reviews by ascending time order. Removes
    "time" field as it has been included within other fields.

    Args:
        rate_beer: The list of ratings.

    Returns:
        The list of reviews with an 'id' key-value of the reviews and the "time"
        field removed.
    """
    sorted_reviews = sorted(rate_beer, key=lambda review: int(review["review"]["time"]))
    for i in range(len(sorted_reviews)):
        sorted_reviews[i]["id"] = f"{i}"
        del sorted_reviews[i]["review"]["time"]
    return sorted_reviews


def _add_triples(key, list_of_head_rel_tail, review_id, review):
    """
    Adds a triple in the format of <entity1, relationship, entity2>
    Args:
        key:
        list_of_head_rel_tail: The list to add the triple to.
        review_id: The Id of the review to add to.
        review: The review.

    Returns:
    """
    for field, value in review[key].items():
        if field == "style":
            beer_id = "bee" + review["beer"]["beerId"]
            list_of_head_rel_tail.append([beer_id, field, value])
            continue
        list_of_head_rel_tail.append([review_id, field, field[:3] + value])
    return list_of_head_rel_tail


def split_and_strip_line(line):
    cleaned_line = line.rstrip()
    return cleaned_line.split("\t")
