import numpy as np
import torch
from torchvision import transforms, models
from PIL import Image
import json
import argparse

# Creates Argument Parser object named parser
parser = argparse.ArgumentParser()

parser.add_argument('image_path', type = str, 
                    help = 'Provide the path to a singe image (required)')
parser.add_argument('save_path', type = str, 
                    help = 'Provide the path to the file of the trained model (required)')

parser.add_argument('--category_names', type = str,
                    help = 'Use a mapping of categories to real names')
parser.add_argument('--top_k', type = int, default = 5,
                    help = 'Return top K most likely classes. Default value is 5')
# GPU
parser.add_argument('--gpu', action='store_true',
                    help = "Add to activate CUDA")

args = parser.parse_args()

if args.gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# Load the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    if checkpoint['arch'] == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif checkpoint['arch'] == 'vgg13':
        model = models.vgg13(pretrained = True)
    else:
        raise ValueError('Model arch error.')

    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    
    return model


checkpoint_path = args.save_path
model = load_checkpoint(checkpoint_path)
model.to(device)

### ------------------------------------------------------------




def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    imagen = Image.open(image)
    imported_image_transform = transforms.Compose([transforms.Resize(255),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                              ])
    
    img = np.array(imported_image_transform(imagen))
    
    return img
    





def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    imagen = process_image(image_path)
    imagen = torch.from_numpy(imagen).type(torch.FloatTensor)
    imagen = imagen.unsqueeze(0)
    imagen = imagen.to(device)

    model.eval()

    with torch.no_grad():
        output = model.forward(imagen)
    output_prob = torch.exp(output)
    probability, indexes = output_prob.topk(topk)
    probability = probability.to('cpu').numpy().tolist()[0]
    indexes = indexes.to('cpu').numpy().tolist()[0]

    mapping = {val: key for key, val in model.class_to_idx.items()}
    classes = [mapping[item] for item in indexes]

    return probability, classes
 

# Make the prediction
image_path = args.image_path
top_k      = args.top_k

probs, classes = predict(image_path, model, topk=top_k)


# Link the prediction to its category
if args.category_names:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    names = [cat_to_name[key] for key in classes]
    print("Class name:")
    print(names)


print("Class number:")
print(classes)
print("Probability (%):")
for idx, item in enumerate(probs):
    probs[idx] = round(item*100, 2)
print(probs)
