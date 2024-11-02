import wandb
import random
import torch
import numpy as np

wandb.init(project="yacup_ml_conformer_aug_again_roll_volume")

from models.train_module import TrainModule
from utils import initialize_logging, load_config
config = load_config(config_path="./config/config.yaml")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

if "seed" in config:
    set_seed(config["seed"])

initialize_logging(config_path="./config/logging_config.yaml", debug=False)
trainer = TrainModule(config)
trainer.pipeline()
trainer.test()
