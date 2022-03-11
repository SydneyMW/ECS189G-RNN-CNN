import sys
sys.path.append('C:/Users/Sydney/Desktop/ECS 189G/Project')

# from code_.base_class.dataset import dataset
from code_.base_class.evaluate import evaluate
from code_.base_class.method import method
# from code_.base_class.result import result
from code_.base_class.setting import setting
from script.stage_5_script.Dataset_Loader_Node_Classification import Dataset_Loader

from sklearn.metrics import accuracy_score                  # EVALUATE_ACCURACY
import torch                                                # METHOD
import torch.nn as nn                                       # METHOD
import torch.nn.functional as F                             # METHOD
from torch_geometric.data import Data                       # METHOD
from torch_geometric.nn import GCNConv                      # METHOD
from torch_geometric.nn import MessagePassing               # METHOD
from torch_geometric.utils import add_self_loops, degree    # METHOD
import matplotlib.pyplot as plt                             # PLOTTING

### EVALUATE.PY MODIFICATIONS

class Evaluate_Accuracy(evaluate):
    data = None
    
    def evaluate(self):
        print('evaluating performance...')
        return accuracy_score(self.data['true_y'], self.data['pred_y']) # report accuracy metric


### METHOD.PY MODIFICATIONS

class Method(method, torch.nn.Module):
    # https://towardsdatascience.com/program-a-simple-graph-net-in-pytorch-e00b500a642d
    def __init__(self, mName, mDescription, trainingdata):
        method.__init__(self, mName, mDescription)
        torch.nn.Module.__init__(self)
        # super(Method, self).__init__()
        self.device = torch.device("cpu")
        self.conv1 = GCNConv(trainingdata.num_node_features, 200).to(self.device) # modify for dataset sizes
        self.conv2 = GCNConv(200, trainingdata.num_classes).to(self.device)  # modify for dataset sizes

    def forward(self, trainingdata):
        x, edge_index = trainingdata.x, trainingdata.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

    def test(self, testingdata, train=False):
        self.model.eval()
        correct = 0
        pred = self.model(testingdata).max(dim=1)[1]
        if train:
            correct += pred[testingdata.train_mask].eq(testingdata.y[testingdata.train_mask]).sum().item()
            return correct / (len(testingdata.y[testingdata.train_mask]))
        else:
            correct += pred[testingdata.test_mask].eq(testingdata.y[testingdata.test_mask]).sum().item()
            return correct / (len(testingdata.y[testingdata.test_mask]))

    def train(self, trainingdata, plot=True):
        train_accuracies, test_accuracies = list(), list()
        # Adjust learning rate, optimizer, num epochs
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4) # all 3 datasets
        loss_function = nn.CrossEntropyLoss()  # all 3 datasets
        for epoch in range(100): # 100 for Cora, 100 for Citeseer, 300 for Pubmed
                # model.train()
                optimizer.zero_grad()
                # out = model(trainingdata)
                out = self(trainingdata)
                loss = loss_function(out[self.data['train_test_val']['idx_train']], trainingdata.y)
                # loss = F.nll_loss(out[trainingdata.train_mask], trainingdata.y[trainingdata.train_mask])
                loss.backward()
                optimizer.step()

                train_accuracy = self.test(trainingdata, train = True)
                test_accuracy = self.test(trainingdata, train = False)

                train_accuracies.append(train_accuracy)
                test_accuracies.append(test_accuracy)
                print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}'.
                    format(epoch, loss, train_accuracy, test_accuracy))
    
        if plot: # plot loss, training accuracy, testing accuracy over epochs
            plt.plot(train_accuracies, label="Train accuracy")
            plt.plot(test_accuracies, label="Test accuracy")
            # plt.plot(Loss, label = "Training Loss")
            plt.xlabel("# Epoch")
            plt.ylabel("Accuracy")
            # plt.ylabel("Loss")
            plt.legend(loc='upper right')
            plt.show()

    def call_train(self):
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html
        edge_idx = torch.LongTensor(self.data['graph']['edge']).t().contiguous()
        trainingdata = Data(
            x = self.data['graph']['X'].to(self.device),
            edge_index = edge_idx.to('self.device'),
            y = self.data['graph']['y'][self.data['train_test_val']['idx_train']].to(self.device),
            pos = self.data['train_test_val']['idx_train'].to(self.device))
        testingdata = Data(
            x = self.data['graph']['X'].to(self.device),
            edge_index = edge_idx.to(self.device),
            y = self.data['graph']['y'][self.data['train_test_val']['idx_test']].to(self.device),
            pos = self.data['train_test_val']['idx_train'].to(self.device))
        self.model = self(trainingdata).to(device)
        # self.train(trainingdata)

### SETTING.PY MODIFICATIONS

class Setting(setting):
    def load_run_save_evaluate(self):
        self.method.data = self.dataset.load()

        trained = self.method.call_train() 
        self.result.data = trained
        self.evaluate.data = trained

        true_y = self.result.data['true_y'].cpu()
        pred_y = self.result.data['pred_y'].cpu()
        return accuracy_score(true_y, pred_y)

