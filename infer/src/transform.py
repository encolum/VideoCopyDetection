from torchvision import transforms
from PIL import Image

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def sscd_transform(width,height):
    return transforms.Compose(
    [
        transforms.Resize([width, height],interpolation=BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)
def eff_transform(width,height):
    return transforms.Compose([
                transforms.Resize([width, height], interpolation=BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
def vit_transform(width,height):
    return transforms.Compose([
                transforms.Resize([width, height], interpolation=BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])