import cv2
import numpy as np
import matplotlib.pyplot as plt
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.featuremaps = []
        self.gradients = []

        target_layer.register_forward_hook(self.save_featuremaps)
        target_layer.register_backward_hook(self.save_gradients)

    def save_featuremaps(self, module, input, output):
        self.featuremaps.append(output)

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients.append(grad_output[0])

    def get_cam_weights(self, grads):
        # compute the neuron weights (it is the mean of the gradients)
        return np.mean(grads, axis=(1, 2))

    def __call__(self, image, label=None):
        preds = self.model(image)
        self.model.zero_grad()

        if label is None:
            label = preds.argmax(dim=1).item()
        # computing the gradients with backward
        preds[:, label].backward()

        featuremaps = self.featuremaps[-1].cpu().data.numpy()[0, :]
        gradients = self.gradients[-1].cpu().data.numpy()[0, :]

        weights = self.get_cam_weights(gradients)
        cam = np.zeros(featuremaps.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            # multiply the feature map by the neuron weight
            cam += w * featuremaps[i]

        #ReLU because we only want the positive contribution
        cam = np.maximum(cam, 0)
        
        # At this point the cam is a 7x7 matrix, we need to resize it to the original image size (224x224)
        cam = cv2.resize(cam, image.shape[-2:][::-1])
        
        # Normalize the cam to be plotted with a heatmap
        cam_normalized = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        
        return label, cam_normalized

def deprocess_image(image):
    """ take a pytorch tensor (normalized) and return a numpy array which is denormalized (and clipped to 0-1)

    Args:
        image (tensor): _description_

    Returns:
        np array: denormalized image
    """
    image = image.cpu().numpy()
    image = np.squeeze(np.transpose(image[0], (1, 2, 0)))
    image = image * np.array((0.229, 0.224, 0.225)) + \
        np.array((0.485, 0.456, 0.406))  # un-normalize
    image = image.clip(0, 1)
    return image
   
def plot_gradcam(image, dense_cam, filename):
    
    image = deprocess_image(image)
    name_dict = {
        'Original Image': image,
        'GradCAM (DenseNet-121)': apply_mask(image, dense_cam)
    }

    plt.style.use('seaborn-notebook')
    # fig = plt.figure(figsize=(8, 4))

    
    # for i, (name, img) in enumerate(name_dict.items()):
    #     ax = fig.add_subplot(1, 2, i+1, xticks=[], yticks=[])
    #     if i:
    #         img = img[:, :, ::-1]
    #     ax.imshow(img)
    #     ax.set_xlabel(name)

    # fig.suptitle(
    #     'Localization with Gradient based Class Activation Maps', fontsize=14
    # )
    #plt.tight_layout()
    #fig.savefig(filename +'.png')
    #plt.show()
    #plt.close()

    #fig2 = plt.figure()
    plt.imsave("../back/uploads_reason/"+filename, name_dict['GradCAM (DenseNet-121)'][:, :, ::-1])

    
def apply_mask(image, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(image)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)