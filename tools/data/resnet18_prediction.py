import torch
import numpy as np
import torchvision
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import urllib
import os
import pandas as pd
import pdb

# def main():
#     model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
#     model.eval()

#     dataset_root = '/home/jinchoi/src/video-data-aug/data/imagenet'
#     dataset_images = os.path.join(dataset_root, 'images')

#     phases = ['val', 'train']

#     for phase in phases:
#         url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
#         try: urllib.URLopener().retrieve(url, filename)
#         except: urllib.request.urlretrieve(url, filename)

#         input_image = Image.open(filename)
#         preprocess = transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ])
#         input_tensor = preprocess(input_image)
#         input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

#         # move the input and model to GPU for speed if available
#         if torch.cuda.is_available():
#             input_batch = input_batch.to('cuda')
#             model.to('cuda')

#         with torch.no_grad():
#             output = model(input_batch)
#         # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes

#         # print(output[0])
#         # # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
#         # print(torch.nn.functional.softmax(output[0], dim=0))

#         output_npy = torch.nn.functional.softmax(output[0], dim=0).cpu().numpy()
#         output_label = np.argmax(output_npy)
#         print('predicted label: {}'.format(output_label))
#         pdb.set_trace()

#         print('')

def main():
    """
        required keys to read the prediction npy file: 
            1) video id. e.g., 'v_ApplyEyeMakeup_g01_c01'
            2) frame idx, 1-based. e.g., 3
            3) 'logit' for 1000-dim logit, 'prob' for 1000-dim probability
    """

    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
    model.eval()

    # dataset_root = 'data/ucf101/rawframes'
    # listfile_path = '/work/vllab/dataset/UCF101/listfiles/UCF101_frames_list_20201014.csv'
    # output_npy_path = '/work/vllab/dataset/UCF101/resnet18_results/resnet18_prtrnd_preds.npy'
    
    dataset_root = 'data/hmdb51/rawframes'
    # listfile_path = '/work/vllab/dataset/HMDB51/listfiles/HMDB51_frames_list_20201025.csv'
    # output_npy_path = '/work/vllab/dataset/HMDB51/resnet18_results/resnet18_prtrnd_preds.npy'
    listfile_path = '/home/jinchoi/src/rehab/dataset/action/HMDB51/listfiles/HMDB51_frames_list_20201025.csv'
    output_npy_path = '/home/jinchoi/src/rehab/dataset/action/HMDB51/resnet18_results/resnet18_prtrnd_preds.npy'

    dataset_root = 'data/kinetics400/rawframes_train'
    listfile_path = 'data/kinetics400/kinetics400_train_list_rawframes.txt'
    output_npy_path = '/home/jinchoi/src/rehab/dataset/action/kinetics/resnet18_results/resnet18_prtrnd_preds_kinetics100_train.npy'

    df = pd.read_csv(listfile_path, header=None, sep=' ')
        
    #pdb.set_trace()
    output_dict = dict()
    for j,dirname in enumerate(df[0]):
        dirname = '/'.join(dirname.split('/')[-2:])
        
        if 1: # j%100 == 0:
            print("Processing {}, {}/{}".format(dirname, j, len(df[0])))
        
        cur_dir = os.path.join(dataset_root,dirname)
        cur_frames = os.listdir(cur_dir)
        cur_vid = dirname.split('/')[-1][:11]
        for i,filename in enumerate(cur_frames):
            cur_frm_idx = int(filename.split('.')[0].split('_')[1])
            if cur_vid not in output_dict:
                output_dict[cur_vid] = dict()
            filename = os.path.join(cur_dir, filename)
            input_image = Image.open(filename)
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            input_tensor = preprocess(input_image)
            input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

            # move the input and model to GPU for speed if available
            if torch.cuda.is_available():
                input_batch = input_batch.to('cuda')
                model.to('cuda')

            with torch.no_grad():
                output = model(input_batch)
            # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes

            output_npy = torch.nn.functional.softmax(output[0], dim=0).cpu().numpy()

            output_dict[cur_vid][cur_frm_idx] = {'logit': output[0].cpu().numpy(), 'prob': output_npy}

            # output_label = np.argmax(output_npy)
            # print('predicted label: {}'.format(output_label))
    np.save(output_npy_path, output_dict)    

if __name__ == '__main__':
    main()
