import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# Load from Hub 🔥
model = timm.create_model(
    'davit_tiny',
    checkpoint_path=r'./Checkpoint/model_best.pth.tar',
    num_classes=4
)

# Set model to eval mode for inference
model.eval()

# Create Transform
transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

# Get the labels from the model config
labels = ['有害垃圾', '厨余垃圾', '其他垃圾', '可回收垃圾']
top_k = min(len(labels), 5)

# Use your own image file here...
image = Image.open(r'o.png').convert('RGB')

# Process PIL image with transforms and add a batch dimension
x = transform(image).unsqueeze(0)

# Pass inputs to model forward function to get outputs
out = model(x)

# Apply softmax to get predicted probabilities for each class
probabilities = torch.nn.functional.softmax(out[0], dim=0)

# Grab the values and indices of top 5 predicted classes
values, indices = torch.topk(probabilities, top_k)

# Prepare a nice dict of top k predictions
predictions = [
    {"label": labels[i], "score": v.item()}
    for i, v in zip(indices, values)
]
print(predictions)
