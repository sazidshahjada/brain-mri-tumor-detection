import gradio as gr
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 2
class_names = ["No Tumor", "Tumor"]  # adjust if your dataset labels are different

def get_model(model_name, num_classes):
    if model_name == "mobilenet":
        model = models.mobilenet_v2(pretrained=False)
        model.classifier[0] = nn.Dropout(p=0.5)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    elif model_name == "vgg19":
        model = models.vgg19(pretrained=False)
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif model_name == "densenet":
        model = models.densenet121(pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == "resnet":
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "vit":
        model = models.vit_b_16(pretrained=False)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    else:
        raise ValueError("Unknown model name")
    return model.to(device)

# Load Mobilenet weights
model = get_model("mobilenet", num_classes)
model.load_state_dict(torch.load("saved_models/mobilenet_model.pth", map_location=device))
model.eval()

# Preprocessing
img_size = 224
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Prediction Function
def predict(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
    pred_idx = probs.argmax()
    return {class_names[i]: float(probs[i]) for i in range(len(class_names))}

# Gradio Interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="Brain MRI Tumor Detection",
    description="Upload a brain MRI image and the model will predict if it has a tumor or not."
)

iface.launch()
