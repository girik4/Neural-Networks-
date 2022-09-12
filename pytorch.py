import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision



def get_data_loader(training = True):
    """
    TODO: implement this function.

    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    if(training == True):
        data_set = datasets.MNIST('./data', train=True, download=False,
                       transform=custom_transform)
    else:
        data_set = datasets.MNIST('./data', train=False,
                       transform=custom_transform)
    
    loader = torch.utils.data.DataLoader(data_set, batch_size = 50)
    
    return loader



def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    
    model = nn.Sequential(
        
    nn.Flatten(),
    nn.Linear(784,128),
    nn.ReLU(),
    nn.Linear(128,64),
    nn.ReLU(),
    nn.Linear(64,10))
    
    return model




def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    criterion = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()
    
    
    for epoch in range(T):  # loop over the dataset multiple times

        running_loss = 0.0
        acc = 0
        for i, data in enumerate(train_loader, 0):
            #print(data)
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            opt.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()

            # print statistics
            running_loss += loss.item()
            
            n = 0
            for j in range(len(labels)):
                n+= 1
                label = torch.argmax(outputs[j])
                if(label == labels[j]):
                    acc+=1
        
        accuracy = (acc/len(train_loader.dataset)*100)
        accuracy = '{:.2f}'.format(accuracy)
        
        l = (running_loss/len(train_loader.dataset)*100)
        l = '{:.3f}'.format(l)
        
        print("Train Epoch: " + str(epoch)+ " Accuracy: "+str(acc)+"/"+str(len(train_loader.dataset))+ "(" +str(accuracy)+"%) "+ "Loss: "+ str(l))
                    
                
            
            

def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    
    model.eval()
    #criterion = nn.CrossEntropyLoss()
    #opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    with torch.no_grad():
        running_loss = 0.0
        acc = 0
        for i, data in enumerate(train_loader, 0):
            #print(data)
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
           # opt.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
           # loss.backward()
           # opt.step()

            # print statistics
            running_loss += loss.item()
            
            n = 0
            for j in range(len(labels)):
                n+= 1
                label = torch.argmax(outputs[j])
                if(label == labels[j]):
                    acc+=1
        
        accuracy = (acc/len(train_loader.dataset)*100)
        accuracy = '{:.2f}'.format(accuracy)
        
        l = (running_loss/len(train_loader.dataset)*100)
        l = '{:.4f}'.format(l)
        
        if show_loss == False:
            print("Accuracy: " + accuracy +"%")
        else:
            print("Average Loss: "+ l)
            print("Accuracy: " + accuracy +"%")
            
        
    


def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    class_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    
    logits = model(test_images)
    prob = F.softmax(logits, dim=1)
    prob = list(prob[index])  
    prob = sorted(prob, reverse = True)
    for i in prob[:3]:
        print(class_names[prob.index(i)], end = ": ")
        val = i.tolist()*100
        val = '{:.2f}'.format(val)
        print(val + '%')
    


if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()