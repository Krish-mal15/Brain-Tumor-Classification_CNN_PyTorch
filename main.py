import torch
from torchvision.transforms import transforms
from PIL import Image
from train import TumorModel

model = TumorModel(3, 4, 4)
model.load_state_dict(torch.load('Tumor_Model.pth'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load your new image
new_image_path = 'tumor-data/Testing/meningioma/Te-me_0016.jpg'
new_image = Image.open(new_image_path)

input_data = transform(new_image).unsqueeze(0)

with torch.no_grad():
    output = model(input_data)

predicted_class = torch.argmax(output, dim=1).item()

print("Predicted class:", predicted_class)


def label():
    if predicted_class == 0:
        print("Tumor Type: Glioma")
    if predicted_class == 1:
        print("Tumor Type: Meningioma")
    if predicted_class == 2:
        print("No Tumor")
    if predicted_class == 3:
        print("Tumor Type: Pituitary")


label()
