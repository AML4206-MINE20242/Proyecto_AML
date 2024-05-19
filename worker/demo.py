######################################################################################################
# Libraries used in main.py
######################################################################################################
import torch
import torchvision.transforms as transforms
from PIL import Image
from config import model_config
import torch.nn as nn
import torchvision
from gradcam import GradCAM, plot_gradcam
from collections import OrderedDict
#####################################################################################################
# Model
#####################################################################################################

class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121()
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x
#####################################################################################################
# Remove module
#####################################################################################################

def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:] 
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict
#####################################################################################################
# Functions used in main.py
#####################################################################################################

def load_model(model_path,args):
    checkpoint = torch.load(model_path)

    model = DenseNet121(args.num_classes)

    # Remove 'module.' prefix from state dict keys
    model_state_dict = remove_module_prefix(checkpoint['state_dict'])

    # Load the new state dictionary into the model
    model.load_state_dict(model_state_dict)
    model.eval()

    return model

# Function to preprocess the input image
def preprocess_image(image_path,args):
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Assuming same normalization as during training
    ])
    image = image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

# Function to predict classes
def predict_classes(model, image):
    with torch.no_grad():
        output = model(image)
        probabilities = torch.sigmoid(output)
    return probabilities

def print_highest_probability_class(class_names, probabilities):
    highest_prob_idx = probabilities.argmax(dim=1).item()
    print(f"Class: {class_names[highest_prob_idx]}, Highest Probability: {probabilities.flatten()[highest_prob_idx]}")
    return class_names[highest_prob_idx], probabilities.flatten()[highest_prob_idx]

def run_demo(image_path, model_path, class_names,args,input_file):
    model = load_model(model_path,args)
    densenet_model = model.densenet121
    image = preprocess_image(image_path,args)
    probabilities = predict_classes(model, image)
    output_target, probablity = print_highest_probability_class(class_names, probabilities)
    cam_obj = GradCAM(model= densenet_model, target_layer=densenet_model.features[-1])
    _, dense_cam = cam_obj(image, None)
    plot_gradcam(image, dense_cam, input_file)
    return output_target, probablity

###############################################################################
# MAIN
###############################################################################
def classify_image(input_file):
    name_image = "../back/uploads/" + input_file
    args = model_config()
    class_names = args.class_names
    model_path = "m-epoch_FL3.pth.tar"
    output_target, probablity = run_demo(name_image, model_path, class_names,args,input_file)
    return output_target, probablity