from cnn import CNN
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = CNN(120, 120).to(DEVICE)

model.load_state_dict(torch.load("models/2024-12-08 15:05:29_sketch_cnn.pth"))

model.eval()



