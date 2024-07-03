from collections import OrderedDict


from omegaconf import DictConfig

import torch

from model import TransE,test
from utils import preprocess_knowledge_graph

def get_on_fit_config(config: DictConfig):
    """Return function that prepares config to send to clients."""

    def fit_config_fn(server_round: int):
        return {
            "lr": config.lr,
            "momentum": config.momentum,
            "local_epochs": config.local_epochs,
        }

    return fit_config_fn


def get_evaluate_fn(testloader):
    """Define function for global evaluation on the server."""

    def evaluate_fn(server_round: int, parameters, config):


        # train_triplets, entity_to_idx, relation_to_idx = preprocess_knowledge_graph(testloader)
        # model = TransE(483, len(relation_to_idx), 50)
       

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # params_dict = zip(model.state_dict().keys(), parameters)
        # state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        # print(state_dict)
        # model.load_state_dict(state_dict, strict=False)

        #implement evaluation logic if needed
        return 0.0, {"accuracy": 1}
        # Here we evaluate the global model on the test set. Recall that in more
        # realistic settings you'd only do this at the end of your FL experiment
        # you can use the `server_round` input argument to determine if this is the
        # last round. If it's not, then preferably use a global validation set.
        #loss, accuracy = test(model, testloader, device)

        # Report the loss and any other metric (inside a dictionary). In this case
        # we report the global test accuracy.
        #return loss, {"accuracy": accuracy}

    return evaluate_fn
