"""Config used for training and testing the classifier DenseNet121 """

import argparse

def model_config():

    parser = argparse.ArgumentParser(description='PyTorch Brain Tumor Segmentation')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='input batch size for training (default: idk)')
    parser.add_argument('--epochs', type=int, default=15,
                        help='number of epochs to train (default: idk)')
    parser.add_argument('--lr', type=float, default= 0.0001,
                        help='learning rate (default: idk)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='learning rate decay factor (default: 0.5)')
    parser.add_argument('--step-size', type=int, default=2,
                        help='step size for scheduler (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=22,
                        help='random seed (default: 22)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before '
                            'logging training status')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
    parser.add_argument('--img-size', type=int, default=224,
                        help='size of the input image (default: 224)')
    parser.add_argument('--num-classes', type=int, default=5,
                        help='number of classes in the dataset (default: 5)')
    parser.add_argument('--loss', type=str, default='BCEWithLogitsLoss',
                        help='loss function to use (default: BCELoss)')
    parser.add_argument('--eps', type=float, default=1e-08,
                        help='epsilon value to use for numerical stability in BCELoss (default: 1e-8)')
    parser.add_argument('--class-names',nargs="*", type= str, default=['No Finding','Cardiomegaly', 'Edema', 'Pneumothorax', 'Pleural Effusion'],
                        help='names of the classes in the dataset (default: CheXpert class names)')
    parser.add_argument('--com-round', type=int, default=3,
                        help='round of the competition (default: 0)')
    parser.add_argument('--patience', type=int, default=5,
                        help='early stopping patience (default: 5)')
    args = parser.parse_args()

    return args