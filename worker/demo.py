import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as sch
from torch.utils.data import DataLoader
from torchvision import transforms
import logging as log
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import torch.nn.functional as F
import pandas as pd
import sys

sys.path.append('../')
from worker.gradcam import GradCAM, plot_gradcam
from worker.Chexpert import ChexpertDataset
from worker.model import DenseNet121

args = {
    "seed": 22,
    "no_cuda": False,
    "img_size": 384,
    "num_classes": 5,
    "batch_size": 48,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "eps": 1e-08,
}
clase = {
    0: "No Finding",
    1: "Cardiomegaly",
    2: "Edema",
    3: "Pneumothorax",
    4: "Pleural Effusion"
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(args["seed"])

def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:] 
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict
pathFileTest = './CheXpert-v1.0-small/test_mod2.csv'
kwargs = {'num_workers': 2, 'pin_memory': True} if args["no_cuda"] else {}
transform = transforms.Compose([
    transforms.Resize((args["img_size"], args["img_size"])),
    transforms.ToTensor()
])
# Load the model
model = DenseNet121(args["num_classes"])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
nn.init.constant_(model.densenet121.classifier.bias, 0)


def test(model, test_loader, checkpoint, device='cuda'):
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint, map_location=torch.device(device))

    model = DenseNet121(args["num_classes"])
    nn.init.constant_(model.densenet121.classifier.bias, 0)
    optimizer = optim.Adam(model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"], eps=args["eps"]) 
    try:
        model_state_dict = remove_module_prefix(checkpoint['model_state_dict'])
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    except KeyError:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    y_pred = []
    with torch.no_grad():
        for test_data in tqdm(test_loader):
            test_images, label, path = test_data[0].to(device), test_data[1].to(device), test_data[2]
            outputs = model(test_images)
            preds = outputs.argmax(dim=1)
            probs = F.softmax(outputs, dim=1)
            y_pred.extend(preds.cpu().numpy())
    return y_pred[0], probs.cpu().numpy()[0][y_pred][0]

def classify_image(input_file):
    name_image = "../back/uploads/" + input_file
    single_image_dataset = ChexpertDataset(img_path=name_image, transform=transform, inference=True)
    single_image_loader = DataLoader(single_image_dataset, batch_size=1, shuffle=False, **kwargs)
    pred, prob =test(model, single_image_loader, '../worker/best_metric_model.pth', 'cpu')
    cam_obj = GradCAM(model= model.densenet121, target_layer=model.densenet121.features[-1])
    _, dense_cam = cam_obj(single_image_dataset.__getitem__(0)[0].unsqueeze(0).to(device), None)
    plot_gradcam(single_image_dataset.__getitem__(0)[0].unsqueeze(0).to(device), dense_cam, input_file)
    print(clase[pred], prob)
    return clase[pred], prob