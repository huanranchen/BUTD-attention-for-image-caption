from criterions.CIDEr import CIDEr
from DataProcess import get_cider_loader
from model.decoder import Decoder
import torch

decoder = Decoder(device = torch.device('cpu'))
decoder.load_model()
loader, names = get_cider_loader()
cider = CIDEr(loader, decoder, names)
print(names)
cider.load_processed_data()
print(cider.estimate_entire_cider())

