from torch import nn
import torch
from sklearn.metrics import accuracy_score


class Trainer():
    def __init__(self, 
            model, 
            train_dataloader, val_dataloader, test_dataloader):
        
        self.model = model

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        self.loss_fn = nn.BCEWithLogitsLoss()
    
    def train(self, num_epochs, optimizer, create_graph=False):
        train_accs = []
        val_accs = []
        losses = []
        loss_granular = []
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(self.train_dataloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)

                loss = self.loss_fn(outputs.squeeze(), labels)
                loss.backward()
                optimizer.step()

                loss_granular.append(loss.detach().numpy())

                # print statistics
                running_loss += loss.item()
                if i % 1000 == 999:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss.item():.3f}')
                    running_loss = 0.0
            print(f'[Epoch:{epoch + 1}] loss: {running_loss / len(self.train_dataloader):.3f}')
            if create_graph:
                # Predict after each epoch of training
                train_preds, train_Y = self.predict(self.train_dataloader)
                val_preds, val_Y = self.predict(self.val_dataloader)
                
                train_acc = accuracy_score(train_Y, train_preds)
                val_acc = accuracy_score(val_Y, val_preds)

                train_accs.append(train_acc)
                val_accs.append(val_acc)
                losses.append(running_loss)

        return train_accs, val_accs, losses #loss_granular

    def predict(self, dataloader):
        # returns predictions and corresponding labels
        Y = []
        preds = []
        with torch.no_grad():
            for data in dataloader:
                inputs, labels = data
                # calculate outputs by running images through the network
                outputs = self.model(inputs)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                preds.extend(predicted)
                Y.extend(labels)
        return preds, Y


        





