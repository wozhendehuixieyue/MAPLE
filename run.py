import argparse
import wandb
from models.maple import MAPLE
from models.sigma import SIGMA
from models.gru4rec import GRU4Rec
from models.mamba4rec import Mamba4Rec
from models.echomamba4rec import ECHOMamba4Rec
from recbole.model.sequential_recommender import SASRec, BERT4Rec, FEARec
import warnings
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from logging import getLogger
from recbole.data.transform import construct_transform
import sys
from recbole.utils import (
    init_logger,
    init_seed,
    set_color,
    get_flops,
    get_environment,
)


def get_model_class(model_name):
    model_dict = {
        'MAPLE': MAPLE,
        'SIGMA': SIGMA,
        'GRU4Rec': GRU4Rec,
        'BERT4Rec': BERT4Rec,
        'SASRec': SASRec,
        'FEARec': FEARec,
        'Mamba4Rec': Mamba4Rec,
        'ECHOMamba4Rec': ECHOMamba4Rec,
    }

    if model_name in model_dict:
        return model_dict[model_name]

    try:
        import recbole.model.sequential_recommender as seq_models
        return getattr(seq_models, model_name)
    except AttributeError:
        raise ValueError(f"Model '{model_name}' not found. Please add it to 'model_dict' or check the spelling.")


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='MambaRec', help='Name of the model to run.')

    args, unknown = parser.parse_known_args()
    model_class = get_model_class(args.model)
    kwargs_dict = {}

    config = Config(model=model_class, config_file_list=['config-ori.yaml'], config_dict=kwargs_dict)
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = model_class(config, train_data.dataset).to(config['device'])
    logger.info(model)

    transform = construct_transform(config)
    flops = get_flops(model, dataset, config["device"], logger, transform)
    logger.info(set_color("FLOPs", "blue") + f": {flops}")

    # trainer loading and initialization
    trainer = Trainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, show_progress=config["show_progress"]
    )
    # model evaluation
    test_result = trainer.evaluate(
        test_data, show_progress=config["show_progress"]
    )

    environment_tb = get_environment(config)
    logger.info(
        "The running environment of this training is as follows:\n"
        + environment_tb.draw()
    )

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")

    if wandb.run is not None:
        wandb.finish()
