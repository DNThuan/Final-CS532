from flask import Flask, render_template, request
import torch
import torchvision
from torchvision import transforms
import json
from PIL import Image
import os
import sys
import numpy as np

device = ('cuda' if torch.cuda.is_available() else 'cpu')

webapp_root = "webapp"
static_dir = os.path.join(webapp_root, "static")
template_dir = os.path.join(webapp_root, "templates")


app = Flask(__name__,  static_folder=static_dir,template_folder=template_dir)

model_path = os.path.join(os.getcwd(),"src","models", "model.pth")
model = torch.load(model_path)


def predict(model, image):
    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_SDEV = [0.229, 0.224, 0.225]

    data_transforms =  transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(IMG_MEAN, IMG_SDEV)])
        
    image = data_transforms(image)
    img_tensor = torch.from_numpy(np.expand_dims(image, axis=0)).type(torch.FloatTensor)
    img_tensor = img_tensor.to(device)

    model.eval()
    with torch.no_grad():
        output = torch.exp(model(img_tensor))
        _, output_class = torch.max(output, 1)
    

    return output_class

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        try:
            image = request.files['image_submit']
            image_path = os.path.join(os.getcwd(),"webapp","static","images", image.filename)
            image.save(image_path)

            load_image = Image.open(image_path)

            with open( os.path.join(os.getcwd(),"src","models", "class_to_idx.json")) as f:
                class_to_idx = json.load(f)
            pos = list(class_to_idx.values()).index(predict(model, load_image))
            result = list(class_to_idx.keys())[pos]
            
            print(image_path)

            return render_template("index.html",prediction =  result, img_path = "./static/images/"+image.filename)
        except:
            return {"erro":500}
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000, debug=True)