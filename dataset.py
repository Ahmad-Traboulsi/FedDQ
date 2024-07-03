import pandas as pd
import torch
from torch.utils.data import random_split, DataLoader

def get_data():
    data = pd.read_csv('KG.csv').iloc[:,:3]
    trainset = data.sample(frac=0.8, random_state=42)  # 80% of the data for training
    testset = data.drop(trainset.index)  # Remaining 20% for testing
    return trainset.values.tolist(), testset.values.tolist()

def prepare_dataset(num_partitions: int, batch_size: int, val_ratio: float = 0.1):
    trainset, testset = get_data()

    # split trainset into `num_partitions` trainsets (one per client)
    # figure out number of training examples per partition
    num_rows = len(trainset) // num_partitions

    # a list of partition lenghts (all partitions are of equal size)
    partition_len = [num_rows] * num_partitions

    #split randomly. This returns a list of trainsets, each with `num_rows` training examples
    trainsets = random_split(
        trainset, partition_len, torch.Generator().manual_seed(2023)
    )
    # create dataloaders with train+val support
    trainloaders = []
    valloaders = []
    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(
            trainset_, [num_train, num_val], torch.Generator().manual_seed(1)
        )

        # construct data loaders and append to their respective list.
        # In this way, the i-th client will get the i-th element in the trainloaders list and the i-th element in the valloaders list
        trainloaders.append(
            DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2)
        )
        valloaders.append(
            DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2)
        )

    # We leave the test set intact (i.e. we don't partition it)
    # This test set will be left on the server side and we'll be used to evaluate the
    # performance of the global model after each round.
    # Please note that a more realistic setting would instead use a validation set on the server for
    # this purpose and only use the testset after the final round.
    # Also, in some settings (specially outside simulation) it might not be feasible to construct a validation
    # set on the server side, therefore evaluating the global model can only be done by the clients. (see the comment
    # in main.py above the strategy definition for more details on this)   
    testloader = DataLoader(testset, batch_size=128)

    return trainloaders, valloaders, testloader