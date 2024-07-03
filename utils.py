from colorama import init as colorama_init
from colorama import Fore, Back
from colorama import Style

def preprocess_knowledge_graph(kg):
    entity_to_idx = {}
    relation_to_idx = {}
    triplets = []

    for  row in kg.dataset:
        head, relation, tail = row[0], row[2], row[1]
        # Assign index to entity if not already assigned
        if head not in entity_to_idx:
            entity_to_idx[head] = len(entity_to_idx)
        if tail not in entity_to_idx:
            entity_to_idx[tail] = len(entity_to_idx)
        # Assign index to relation if not already assigned
        if relation not in relation_to_idx:
            relation_to_idx[relation] = len(relation_to_idx)
        triplets.append((entity_to_idx[head], relation_to_idx[relation], entity_to_idx[tail]))

    print(f'{Fore.WHITE}{Back.RED}Finished Pre-processing KG{Style.RESET_ALL}')
    return triplets, entity_to_idx, relation_to_idx