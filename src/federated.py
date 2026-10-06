import torch
import torch.nn as nn
from model import PneumoniaCNN
from prep_data import train_dataset, test_loader
from torch.utils.data import random_split, DataLoader


# FedAvg: averages the same layer's weights across all clients, one layer at a time.
# This is the core "federated averaging" algorithm - combines each client's local
# learning into one improved shared model, without ever seeing their raw data.
def average_weights(weights_list):
    avg_weights = weights_list[0]  # start with client 1's weights as a template
    for key in avg_weights:        # loop through each layer name (e.g. 'conv1.weight')
        for i in range(1, len(weights_list)):  # add the same layer from clients 2, 3, ...
            avg_weights[key] += weights_list[i][key]
        avg_weights[key] = avg_weights[key] / len(weights_list)  # divide by client count = average
    return avg_weights


# Simulate 3 separate "hospitals" by splitting the training data into 3 chunks.
# In real FL, each client's data never leaves their own system - this mirrors that
# by giving each client its own slice, with no overlap between them.
client_data = random_split(train_dataset, [1739, 1739, 1738])

# Each client needs its own DataLoader to serve its slice in batches during local training
client_loaders = []
for data in client_data:
    loader = DataLoader(data, batch_size=32, shuffle=True)
    client_loaders.append(loader)

global_model = PneumoniaCNN()   # the shared model that gets improved each round
num_rounds = 3                  # how many send -> train locally -> average cycles to run
local_epochs = 2                # how many epochs each client trains on its own data, per round

for round in range(num_rounds):
    print(f'----- Round {round+1}/{num_rounds} ------')

    local_weights = []   # collects each client's trained weights this round, for averaging after

    for client_id in range(3):
        # every client starts this round from the SAME current global weights -
        # required for averaging afterward to make sense (see notes/discussion)
        client_model = PneumoniaCNN()
        client_model.load_state_dict(global_model.state_dict())

        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(client_model.parameters(), lr=0.001)

        for epoch in range(local_epochs):
            for images, labels in client_loaders[client_id]:
                # MALICIOUS CLIENT SIMULATION: client 3 (client_id == 2) trains on
                # deliberately flipped labels (label-flipping attack), so it learns
                # the opposite of the correct patterns and sends back corrupted weights
                if client_id == 2:
                    labels = 1 - labels

                optimizer.zero_grad()
                outputs = client_model(images)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

        print(f'Client {client_id+1} finished local training, loss: {loss.item():.4f}')
        local_weights.append(client_model.state_dict())  # save this client's weights to average later

    # once ALL 3 clients finish, average their weights into one improved global model
    averaged_weights = average_weights(local_weights)
    global_model.load_state_dict(averaged_weights)
    print(f'Round {round+1} complete - global model updated')


# Evaluate the FINAL global model (after all rounds + averaging) on the test set -
# same evaluation logic as evaluate.py, applied here to the federated result instead
global_model.eval()

correct = 0
total = 0
normal_predicted = 0
pneumonia_predicted = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = global_model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        normal_predicted += (predicted == 0).sum().item()
        pneumonia_predicted += (predicted == 1).sum().item()

accuracy = 100 * correct / total
print(f'Federated Test Accuracy: {accuracy:.2f}%')
print(f'Predicted Normal: {normal_predicted}')
print(f'Predicted Pneumonia: {pneumonia_predicted}')

torch.save(global_model.state_dict(), 'model_federated.pth')
print('Federated model saved')