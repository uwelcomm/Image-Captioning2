
from PIL import Image

from torchvision import transforms
from our_models import TrainingModel

normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

transform = transforms.Compose([
    transforms.Resize((384,384),interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    normalize,
])     
# model = TrainingModel.load_from_checkpoint('/kaggle/input/captioning-trainer-ckpt/Image-Captioning/checkpoints/best-model.ckpt',config=config)
# model = TrainingModel(config=config)
model = TrainingModel()
model.eval()

config=model.config
# !pip install happytransformer
from happytransformer import HappyTextToText, TTSettings
happy_tt = HappyTextToText("T5", config['checkpoint']+'corrector_ckpt')


args = TTSettings(num_beams=5, min_length=1)

def predict(img_path):
    image = Image.open(img_path).convert('RGB') 
    image = transform(image)
    image=image.unsqueeze(dim=0)
    model.eval()
    visual_caption=model.model.generate(image, sample=False, num_beams=config['num_beams'], max_length=config['max_length'], 
                              min_length=config['min_length'])[0]
    result = happy_tt.generate_text("grammar: "+visual_caption, args=args)
    return result.text
