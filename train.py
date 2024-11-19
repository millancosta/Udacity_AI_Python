import torch
import json
import argparse
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict

# Argument parser
parser = argparse.ArgumentParser()

parser.add_argument('data_dir', type = str,
                    help = 'Provide the data directory, mandatory')
parser.add_argument('--save_dir', type = str, default = './',
                    help = 'Provide the save directory')
parser.add_argument('--arch', type = str, default = 'densenet121',
                    help = 'densenet121 or vgg13')
# hyperparameters
parser.add_argument('--learning_rate', type = float, default = 0.001,
                    help = 'Learning rate, default value 0.001')
parser.add_argument('--hidden_units', type = int, default = 512,
                    help = 'Number of hidden units. Default value is 512')
parser.add_argument('--epochs', type = int, default = 20,
                    help = 'Number of epochs')
# GPU
parser.add_argument('--gpu', action='store_true',
                    help = "Add to activate CUDA")
parser.add_argument('--savefile', type=str, default='checkpoint.pth')

#setting values data loading
args = parser.parse_args()

if args.gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Data is loading, please wait...")

data_dir  = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir  = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(180),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(p=0.5),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

# no need to perform randomization on validation/test samples; only need to normalize
val_test_data_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])


# Load the datasets
image_traindatasets = datasets.ImageFolder(train_dir, transform = train_transforms)
image_valdatasets = datasets.ImageFolder(valid_dir, transform = val_test_data_transforms)
image_testdatasets  = datasets.ImageFolder(test_dir,  transform = val_test_data_transforms)

# Create the iterables
train_dataloaders = torch.utils.data.DataLoader(image_traindatasets, batch_size = 64, shuffle = True)
val_dataloaders = torch.utils.data.DataLoader(image_valdatasets, batch_size = 64, shuffle=True)
test_dataloaders  = torch.utils.data.DataLoader(image_testdatasets,  batch_size = 64, shuffle=True)


# Obtain labels from the category file
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Build the model
print("Model is being built")

layers        = args.hidden_units
learning_rate = args.learning_rate

if args.arch == 'densenet121':
    model = models.densenet121(pretrained=True)
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(1024, layers)),
                              ('relu', nn.ReLU()),
                              ('dropout', nn.Dropout(0.2)),
                              ('fc2', nn.Linear(layers, len(cat_to_name))),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
elif args.arch == 'vgg13':
    model = models.vgg13(pretrained = True)
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, layers)),
                              ('relu', nn.ReLU()),
                              ('dropout', nn.Dropout(0.2)),
                              ('fc2', nn.Linear(layers, len(cat_to_name))),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
else:
    raise ValueError('Model arch error.')

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
model.to(device)

# Train the model
print("The model is being trained")

epochs = args.epochs
steps  = 0
train_losses = []
test_losses = []
for e in range(epochs):
    
    print(f"Epoch: {e}")
    count=0
    running_loss = 0
    for images, labels in train_dataloaders:  
        steps += 1
        print(count)
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # Forward pass through the model to get features
        log_ps = model.forward(images) 
        
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        count += 1
    # Validation phase
    test_loss = 0
    accuracy = 0
    model.eval()
    
    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():
        for images, labels in val_dataloaders:  
            images, labels = images.to(device), labels.to(device)
            log_ps = model.forward(images)  # Get log probabilities from the model
            test_loss += criterion(log_ps, labels).item()
            
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    
    train_losses.append(running_loss / len(train_dataloaders))
    test_losses.append(test_loss / len(val_dataloaders))
    model.train()
    print("Epoch: {}/{}.. ".format(e+1, epochs),
          "Training Loss: {:.3f}.. ".format(running_loss / len(train_dataloaders)),
          "Test Loss: {:.3f}.. ".format(test_loss / len(val_dataloaders)),
          "Test Accuracy: {:.3f}".format(accuracy / len(val_dataloaders)))
 


# Test the model
print("The model is being tested")

model.to(device)

accuracy = 0
model.eval()

with torch.no_grad():
    
    for images, labels in test_dataloaders:

        images, labels = images.to(device), labels.to(device)
        logps = model.forward(images)
        
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

print(f"Test accuracy: {accuracy/len(test_dataloaders):.3f}")

      
model.train()

# Save the model
model.class_to_idx = image_traindatasets.class_to_idx

model.to('cpu')

checkpoint = {'class_to_idx': model.class_to_idx,
              'model_state_dict': model.state_dict(),
              'classifier': model.classifier,
              'arch': args.arch
             }

torch.save(checkpoint, args.savefile)
print("Model saved")
