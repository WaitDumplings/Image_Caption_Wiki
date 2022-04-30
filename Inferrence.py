import torch
import json
from utils import caption_image_beam_search, visualize_att
from model import Encoder,Attention, DecoderWithAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Predict
# Model Pretrained weight path:
# https://drive.google.com/file/d/1-eI3szeX-5__2B7p_Upq_ObmgPJmBIU6/view?usp=sharing
model_path ='./BestWiki.pth.tar'
# word_map_fixed path (generated from COCO_Wiki.json)
word_map = './COCO_Wiki_fixed.json'
# image path
img_path = './Tour bus being used in France.JPG'

beam_size = 3
# Load model
checkpoint = torch.load(model_path, map_location=str(device))
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()

# Load word map (word2ix)
with open(word_map, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

# Encode, decode with attention and beam search
seq, alphas = caption_image_beam_search(encoder, decoder, img_path, word_map, beam_size)
alphas = torch.FloatTensor(alphas)

# Visualize caption and attention of best sequence
visualize_att(img_path, seq, alphas, rev_word_map, True)