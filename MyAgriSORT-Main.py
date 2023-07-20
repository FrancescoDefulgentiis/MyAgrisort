import cv2
import torch
import argparse
from tracker.tracker import Tracker, bbox_to_meas, meas_to_mot
from tools.visualizer import Visualizer
import tools.data_manager as DataUtils
import time
from tqdm import tqdm
import os
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.set_device(0)
else:
    device = torch.device('cpu')

def main(opt):
    #stuff that i dont know why is here
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3149, 0.2998, 0.2792], std=[0.1808, 0.1984, 0.2022])
    ])

    root_dir = 'da decidere'#the directory of the
    model_path = '../NN/models/resnet18_new32.pt'    

    #full dataset containing normalized images
    dataset = GrapeDataset100(root_dir, transform=transform)
    batch_size = 32
    data_loader= DataLoader(dataset, batch_size=batch_size, shuffle=False)

    #pre-trained model
    model = torch.load(model_path)
    #remove latest layers fc e logsoftmax
    model.fc = model.fc[:2]
    model.to(device)
    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad(), tqdm(total=len(data_loader), desc='Calcolo gli embeddings', unit='batch') as progress_bar:
        for frame, labels in data_loader:
            images = images.to(device) 
            labels = labels.to(device)

            embeddings = model(images)

            all_embeddings.append(embeddings)
            all_labels.append(labels)

            progress_bar.update(1)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)


    torch.save({'embeddings': all_embeddings, 'labels': all_labels}, 'embeddings/outputs.pth')

    # If visualizer, initialize visualizer
    print("visuaize: ", opt.visualize)
    #if opt.visualize:
        #create a for loop cycling on the frames of the video with drawn bboxes

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
