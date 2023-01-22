from pathlib import Path

import torch
from pykeen.triples import TriplesFactory

from clustering.rate_beer_loader import RateBeerLoaderPykeen



def test_get_rate_beer_plain_no_checkpoint():
    """
    This will test the Pykeen rate beer loader to ensure it can load the data
    in the format needed.
    """
    torch.manual_seed(100)
    beer_location = Path(__file__).parent.joinpath("ratebeer_test_data.txt").absolute()

    checkpoint_name = "test_pykeen_plain_checkpoint.pt"
    loader = RateBeerLoaderPykeen(beer_location, checkpoint_name)
    rate_beer_res = loader.get_rate_beer()
    _run_training(rate_beer_res, checkpoint_name=checkpoint_name)


def _run_training(training_data, checkpoint_name):
    from pykeen.pipeline import pipeline
    from pykeen.models import TransE

    train_transe, test_transe, eval_transe = training_data.split([0.8, 0.1, 0.1])
    transe_model_pipeline_results = pipeline(training=train_transe, testing=test_transe, validation=eval_transe,
                                             training_loop='sLCWA', model=TransE,
                                             model_kwargs={"embedding_dim": 50},
                                             training_kwargs=dict(checkpoint_name=checkpoint_name,
                                                                  checkpoint_frequency=1, num_epochs=200),
                                             stopper="early",
                                             stopper_kwargs=dict(frequency=2, patience=2, relative_delta=0.002,
                                                                 metric="mean_reciprocal_rank"))
    return transe_model_pipeline_results

def test_loads_previous_values():
    """
    Tests that the pipeline will load again and that the loaded model is not `None`.
    """
    from pykeen.constants import PYKEEN_CHECKPOINTS
    torch.manual_seed(100)
    previous_checkpoint = PYKEEN_CHECKPOINTS.joinpath("torch_pipeline_checkpoints.pt")
    loaded = torch.load(previous_checkpoint)
    entity_to_id = loaded["entity_to_id_dict"]
    relationship_to_id = loaded["relation_to_id_dict"]