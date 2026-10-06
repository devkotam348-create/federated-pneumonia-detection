import torch
import torch.nn as nn
from model import PneumoniaCNN
from prep_data import train_dataset, test_loader
from torch.utils.data import random_split, DataLoader


def average_weights(weights_list):
    avg_weights = weights_list[0]
    for key in avg_weights:
        for i in range(1, len(weights_list)):
            avg_weights[key] += weights_list[i][key]
        avg_weights[key] = avg_weights[key] / len(weights_list)
    return avg_weights


client_data = random_split(train_dataset, [1739, 1739, 1738])

client_loaders = []
for data in client_data:
    loader = DataLoader(data, batch_size=32, shuffle=True)
    client_loaders.append(loader)

global_model = PneumoniaCNN()
num_rounds = 3
local_epochs = 2

for round in range(num_rounds):
    print(f'----- Round {round+1}/{num_rounds} ------')

    local_weights = []

    for client_id in range(3):
        client_model = PneumoniaCNN()
        client_model.load_state_dict(global_model.state_dict())

        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(client_model.parameters(), lr=0.001)

        for epoch in range(local_epochs):
            for images, labels in client_loaders[client_id]:
                if client_id == 2:
                    labels = 1 - labels
                
                optimizer.zero_grad()
                outputs = client_model(images)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

        print(f'Client {client_id+1} finished local training, loss: {loss.item():.4f}')
        local_weights.append(client_model.state_dict())

    averaged_weights = average_weights(local_weights)
    global_model.load_state_dict(averaged_weights)
    print(f'Round {round+1} complete - global model updated')


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