from collections import OrderedDict
from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar
from colorama import init as colorama_init
from colorama import Fore, Back
from colorama import Style
from utils import preprocess_knowledge_graph


import torch
import flwr as fl

from model import TransE, train




class FlowerClient(fl.client.NumPyClient):
    """Define a Flower Client."""

    def __init__(self, trainloader, vallodaer, embedding_dim) -> None:
        super().__init__()

        # the dataloaders that point to the data associated to this client
        self.trainloader = trainloader
        self.valloader = vallodaer

        train_triplets, entity_to_idx, relation_to_idx = preprocess_knowledge_graph(trainloader)

        val_triplets = preprocess_knowledge_graph(vallodaer)
        
        self.train_triplets = train_triplets
        self.val_triplets = val_triplets


        # a model that is randomly initialised at first
        self.model = TransE(len(entity_to_idx), len(relation_to_idx), embedding_dim)
        # figure out if this client has access to GPU support or not
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    """set_parameters are usually used to receive parameters and apply them to the local model."""
    def set_parameters(self, parameters):
        for param, val in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(val)
    
    """Extract model parameters and return them as a list of numpy arrays."""
    def get_parameters(self, config):
        return [val.cpu().detach().numpy() for val in self.model.parameters()]
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.model, self.train_data, epochs=config["epochs"], lr=config["lr"])
        return self.get_parameters(), len(self.train_data), {}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss = 0.0  # Implement evaluation logic if needed
        return float(loss), len(self.train_data), {}   

def generate_client_fn(trainloaders, valloaders, embedding_dim):
    """Return a function that can be used by the VirtualClientEngine.

    to spawn a FlowerClient with client id `cid`.
    """
    def client_fn(cid: str):
        # This function will be called internally by the VirtualClientEngine
        # Each time the cid-th client is told to participate in the FL
        # simulation (whether it is for doing fit() or evaluate())

        # Returns a normal FLowerClient that will use the cid-th train/val
        # dataloaders as it's local data.
        return FlowerClient(
            trainloader=trainloaders[int(cid)],
            vallodaer=valloaders[int(cid)],
            embedding_dim=embedding_dim
        )

    # return the function to spawn client
    return client_fn