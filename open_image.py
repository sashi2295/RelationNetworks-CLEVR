from PIL import Image

img_file = "/home/iki/sashi/robotkoop/CLEVR_v1.0/images/train/CLEVR_train_000000.png"

image = Image.open(img_file).convert('RGB')
