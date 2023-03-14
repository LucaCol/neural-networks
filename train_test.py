from tqdm.notebook import tqdm
import torch

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# training function
def train(model, trainloader, optimizer, criterion):
    # model.train() is a kind of switch for some specific layers/parts of the model that behave differently 
    # during training time.(For example, Dropouts Layers, BatchNorm Layers, etc.)
    model.train()
    
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        
        # from each batch retrieve the images and labels
        image, labels = data
        #labels = labels - 1
        image = image.to(device)
        labels = labels.to(device)
        
        # zeros the gradient of the previous iteration
        optimizer.zero_grad()
        
        # forward pass
        outputs = model(image)
        
        # calculate the loss
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        
        # calculate the accuracy
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        
        # apply the gradient and update the weights
        loss.backward()
        optimizer.step()
    
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))

    return epoch_loss, epoch_acc


# test
def test(model, testloader, criterion):
    # model.eval() is a kind of switch for some specific layers/parts of the model that behave differently 
    # during validation time.(For example, Dropouts Layers, BatchNorm Layers, etc.)
    # combine with torch.no_grad() to turn of the gradient computation
    model.eval()
    
    # we need two lists to keep track of class-wise accuracy
    class_correct = list(0. for i in range(10+1))
    class_total = list(0. for i in range(10+1))
    
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            # from each batch retrieve the images and labels
            image, labels = data
            #labels = labels - 1
            image = image.to(device)
            labels = labels.to(device)
            
            # forward pass
            outputs = model(image)
            
            # calculate the loss
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            
            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()

            # calculate the accuracy for each class
            """
            correct  = (preds == labels).squeeze()
            for i in range(len(preds)):
                label = labels[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1
            """
        
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))

    return epoch_loss, epoch_acc
"""
    # print the accuracy for each class after evey epoch
    # the values should increase as the training goes on
    print('\n')
    for i in range(183):
        print(f"Accuracy of digit {i+1}: {100*class_correct[i]/class_total[i]}")
"""