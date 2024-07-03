import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(TransE, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)

    def forward(self, head, relation, tail):
        head_emb = self.entity_embeddings(head)
        relation_emb = self.relation_embeddings(relation)
        tail_emb = self.entity_embeddings(tail)
        score = torch.norm(head_emb + relation_emb - tail_emb, p=1, dim=1)
        return score
    
def train(model, triplets, epochs=100, lr=0.001):
    criterion = nn.MarginRankingLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    target = torch.tensor([-1], dtype=torch.float)

    for epoch in range(epochs):
        total_loss = 0
        for triplet in triplets:
            head = torch.tensor([triplet[0]], dtype=torch.long)
            relation = torch.tensor([triplet[1]], dtype=torch.long)
            tail = torch.tensor([triplet[2]], dtype=torch.long)

            optimizer.zero_grad()
            pos_score = model(head, relation, tail)
            neg_tail = torch.randint(0, model.entity_embeddings.num_embeddings, tail.size())
            neg_score = model(head, relation, neg_tail)

            loss = criterion(pos_score, neg_score, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss/len(triplets)}')



def test(net, testloader, device: str):
    """Validate the network on the entire test set.

    and report loss and accuracy.
    """
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    net.to(device)
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy
