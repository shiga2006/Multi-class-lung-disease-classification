import torch
from torchvision import transforms
from PIL import Image
import os

from Train_5class import SimpleCNN, ProtoNet

encoder = SimpleCNN(out_dim=64)
model = ProtoNet(encoder)
model.load_state_dict(torch.load("best_protonet_xray_long[5class].pth", map_location='cpu'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

support_dir = "support"
class_names = [
    "viral pneumonia",
    "bacterial pneumonia",
    "normal",
    "covid",
    "tuberculosis"
]
support_images = []
support_labels = []

for i, cls in enumerate(class_names):
    found = False
    for ext in ["jpg", "jpeg", "png"]:
        img_path = os.path.join(support_dir, f"{cls}.{ext}")
        if os.path.isfile(img_path):
            img = Image.open(img_path).convert("RGB")
            img = transform(img)
            support_images.append(img)
            support_labels.append(i)
            found = True
            break
    if not found:
        raise FileNotFoundError(
            f"No support image found for class '{cls}' with extension jpg/jpeg/png in {support_dir}"
        )
support_images = torch.stack(support_images)
support_labels = torch.tensor(support_labels)

def get_explanation_and_first_aid(prediction):
    explanations = {
        "normal": (
            "Your chest X-ray appears normal. No radiological signs of pneumonia, COVID-19, or tuberculosis."
        ),
        "viral pneumonia": (
            "Your chest X-ray suggests viral pneumonia. Consult a healthcare provider for confirmation and treatment.\n"
            "First Aid: Rest, hydrate, monitor symptoms, and seek care if breathing worsens."
        ),
        "bacterial pneumonia": (
            "Your chest X-ray suggests bacterial pneumonia. Antibiotics may be required. Consult a healthcare provider promptly.\n"
            "First Aid: Rest, hydrate, monitor symptoms, and seek care if breathing worsens."
        ),
        "covid": (
            "Your chest X-ray shows signs suggestive of COVID-19 pneumonia. Confirm with a COVID-19 test and follow local health guidelines.\n"
            "First Aid: Isolate, monitor oxygen, rest, hydrate, and seek emergency care if symptoms worsen."
        ),
        "tuberculosis": (
            "Your chest X-ray suggests tuberculosis. TB is a serious infection and requires medical treatment.\n"
            "First Aid: Isolate, wear a mask, and consult a healthcare provider for further testing and treatment."
        ),
    }
    return explanations.get(prediction, "No explanation available.")

def classify(image_path):
    print("Inside classify, image_path:", image_path)
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)
    print("Image transformed:", img.shape)
    logits = model(support_images, support_labels, img)
    pred = torch.argmax(logits, dim=1).item()
    print("Predicted class index:", pred)
    prediction = class_names[pred]
    explanation = get_explanation_and_first_aid(prediction)
    return prediction, explanation