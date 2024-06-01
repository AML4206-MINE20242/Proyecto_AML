import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as sch
from config import model_config
from torch.utils.data import DataLoader
from torchvision import transforms
from Chexpert import ChexpertDataset
from model import DenseNet121, EfficientNetV2, VisionTransformer
import logging as log
import time
import numpy as np
from barbar import Bar
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
#from graphic import plot_roc_curve
from sklearn.metrics import classification_report
from monai.metrics import ROCAUCMetric
from monai.transforms import Activations, AsDiscrete, Compose
from monai.data import decollate_batch
import os
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from collections import OrderedDict
from torch.nn import NLLLoss

# Set the logger
log.basicConfig(filename='normal_chexpert_training.log', level=log.INFO, format='%(asctime)s %(levelname)s %(message)s')
log.info('Started')
# Load the configuration of params
args = model_config()

def log_config(args):
    logger = log.getLogger(__name__)
    logger.info("Configuration:")
    for arg, value in sorted(vars(args).items()):
        logger.info("Argument %s: %r", arg, value)

log_config(args)

# Check cuda availability
args.cuda = not args.no_cuda and torch.cuda.is_available()
assert args.num_classes == len(args.class_names)

# Set random seed for GPU or CPU
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(args.seed)

# Set arguments for Dataloaders
kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}

# Set main directory of data
pathFileTrain = './CheXpert-v1.0-small/train_mod3.csv'
pathFileValid = './CheXpert-v1.0-small/valid_mod3.csv'
pathFileTest = './CheXpert-v1.0-small/test_mod3.csv'

# Create the dataset loaders
transform = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.ToTensor()
])
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.Resize((args.img_size, args.img_size)),
    transforms.ToTensor()
])

train_dataset = DataLoader(ChexpertDataset(csv_file=pathFileTrain, transform=transform_train),
                           batch_size=args.batch_size, shuffle=True, **kwargs)
valid_dataset = DataLoader(ChexpertDataset(csv_file=pathFileValid, transform=transform),
                           batch_size=args.batch_size, shuffle=False, **kwargs)
test_dataset = DataLoader(ChexpertDataset(csv_file=pathFileTest, transform=transform),
                          batch_size=args.batch_size, shuffle=False, **kwargs)

# Load the model
model = VisionTransformer(args.num_classes)

device = torch.device('cuda' if args.cuda else 'cpu')
model = model.to(device)
#nn.init.constant_(model.densenet121.classifier.bias, 0)

# Set number of GPUs to use
if args.cuda:
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs.")
        model = torch.nn.DataParallel(model)
        model.cuda()

# Set the optimizer, scheduler, and loss function
def get_loss_function(loss_name):
    if loss_name == 'BCELoss':
        return nn.BCELoss()
    if loss_name == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss()
    if loss_name == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()
    if loss_name == 'NLLLoss':
        return NLLLoss()

loss = get_loss_function(args.loss).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=args.eps)
scheduler = sch.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

# Model Weights save function
def save_model(model, optimizer, filename):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

# Train, Validate, and Test the model
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, max_epochs, val_interval):
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []

    auc_metric = ROCAUCMetric()
    y_pred_trans = Compose([Activations(softmax=True)])
    y_trans = Compose([AsDiscrete(to_onehot=5)])

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels, path = batch_data[0].to(device), batch_data[1].to(device), batch_data[2]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"{step}/{len(train_loader)}, train_loss: {loss.item():.4f}")
            log.info(f"train_loss: {loss.item():.4f}")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                for val_data in val_loader:
                    val_images, val_labels, val_paths = val_data[0].to(device), val_data[1].to(device), val_data[2]
                    y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                    y = torch.cat([y, val_labels], dim=0)
                y_onehot = [y_trans(i) for i in decollate_batch(y, detach=False)]
                y_pred_act = [y_pred_trans(i) for i in decollate_batch(y_pred)]
                auc_metric(y_pred_act, y_onehot)
                result = auc_metric.aggregate().item()
                auc_metric.reset()
                del y_pred_act, y_onehot
                metric_values.append(result)
                acc_value = torch.eq(y_pred.argmax(dim=1), y)
                acc_metric = acc_value.sum().item() / len(acc_value)
                if result > best_metric:
                    best_metric = result
                    best_metric_epoch = epoch + 1
                    save_model(model, optimizer, os.path.join(f"best_metric_model.pth"))
                    save_model(model, optimizer, os.path.join(f"best_metric_model_{epoch}.pth"))
                    print("saved new best metric model")
                print(f"current epoch: {epoch + 1} current AUC: {result:.4f} current accuracy: {acc_metric:.4f} best AUC: {best_metric:.4f} at epoch: {best_metric_epoch}")
                log.info(f"current epoch: {epoch + 1} current AUC: {result:.4f} current accuracy: {acc_metric:.4f} best AUC: {best_metric:.4f} at epoch: {best_metric_epoch}")
                log.info(f"val_accuracy: {acc_metric:.4f}")
        scheduler.step()

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    return epoch_loss_values, metric_values

def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:] 
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict

def test(model, test_loader, checkpoint, device='cuda'):
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint, map_location=torch.device(device))

    model = VisionTransformer(args.num_classes)
    #nn.init.constant_(model.densenet121.classifier.bias, 0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=args.eps) 
    try:
        model_state_dict = remove_module_prefix(checkpoint['model_state_dict'])
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    except KeyError:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for test_data in tqdm(test_loader):
            test_images, test_labels, test_paths = test_data[0].to(device), test_data[1].to(device), test_data[2]
            outputs = model(test_images)
            preds = outputs.argmax(dim=1)
            y_true.extend(test_labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            

    print("Classification Report:")
    log.info("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["No Finding", "Cardiomegaly", "Edema", "Pneumothorax", "Pleural Effusion"]))
    log.info(classification_report(y_true, y_pred, target_names=["No Finding", "Cardiomegaly", "Edema", "Pneumothorax", "Pleural Effusion"]))
    print("Confusion Matrix:")
    log.info("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    log.info(confusion_matrix(y_true, y_pred))
    save_model(model, optimizer, os.path.join(f"final_model.pth"))
    return y_true, y_pred

train(model, train_dataset, valid_dataset, loss, optimizer, scheduler, device, args.epochs, args.val_inter)
test(model, test_dataset, 'best_metric_model.pth', device)
