"""Config used for training and testing the classifier DenseNet121 """

import argparse

def model_config():

    parser = argparse.ArgumentParser(description='PyTorch Brain Tumor Segmentation')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--img-size', type=int, default=224,
                        help='size of the input image (default: 224)')
    parser.add_argument('--num-classes', type=int, default=5,
                        help='number of classes in the dataset (default: 5)')
    parser.add_argument('--class-names',nargs="*", type= str, default=['No Finding','Cardiomegaly', 'Edema', 'Pneumothorax', 'Pleural Effusion'],
                        help='names of the classes in the dataset (default: CheXpert class names)')
    args = parser.parse_args()

    return args