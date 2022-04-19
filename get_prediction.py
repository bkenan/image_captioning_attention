import torch
import torchvision.transforms as transforms
import io
import glob
from PIL import Image
from get_loader import dataset
from base64 import b64encode

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_caps_from(image_path, final_model):
    # generate the caption
    final_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(image_path, 'rb') as file:
        image_bytes = file.read()
    # transform the image
    features_tensors = transform_image(image_bytes=image_bytes)
    with torch.no_grad():
        features = final_model.encoder(features_tensors.to(device))
        caps, alphas = final_model.decoder.generate_caption(features, vocab=dataset.vocab)
        caption = ' '.join(caps)

    return caption


def get_caps_from_upload(image_path, final_model):
    # generate the caption
    final_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_bytes = image_path.read()
    # transform the image
    features_tensors = transform_image(image_bytes=image_bytes)
    img_base64 = "data:image/png;base64," + b64encode(image_bytes).decode('utf-8')    
    with torch.no_grad():
        features = final_model.encoder(features_tensors.to(device))
        caps, alphas = final_model.decoder.generate_caption(features, vocab=dataset.vocab)
        caption = ' '.join(caps[:-1])
        
    return caption, img_base64


def get_prediction(model, path_to_directory):
    files = glob.glob(path_to_directory+'/*')
    image_with_tags = {}
    for image_file in files:
        image_with_tags[image_file] = get_caps_from(image_file, model)

    return image_with_tags
