""" 
Dataset loader for the Charades-STA dataset 
"""
import math
import os
import csv
import h5py
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
from torchtext import vocab #pip install torchtext==0.8.1

import cv2
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import torchvision
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt

## Custom module dependencies
from . import average_to_fixed_length # "from ." means from init
from core.eval import iou
from core.config import config
from datasets.path import Path as PATH
from datasets.c3d import C3D

class Charades(data.Dataset):

    ## Embedding from torchtext lib
    vocab = vocab.pretrained_aliases["glove.6B.300d"]()
    vocab.itos.extend(['<unk>'])
    vocab.stoi['<unk>'] = vocab.vectors.shape[0]
    vocab.vectors = torch.cat([vocab.vectors, torch.zeros(1, vocab.dim)], dim=0)
    word_embedding = nn.Embedding.from_pretrained(vocab.vectors)
    ## Embedding from file
    '''with open(PATH.embedding_file(),'r') as f:
        vocab = f.readlines()
    vocab = {line.split()[0]:np.asarray(line.split()[1:], "float32") for line in vocab}
    vocab['<unk>'] = np.zeros([1, 300], dtype = "float32")
    word_embedding = nn.Embedding.from_pretrained(torch.Tensor(list(vocab.values())))'''

    def __init__(self, split):
        super(Charades, self).__init__()

        ## Init vars
        #self.vis_input_type = config.DATASET.VIS_INPUT_TYPE
        #self.data_dir = config.DATA_DIR
        self.split = split
        self.distinct_vid = []

        ## Get duration
        self.durations = {}
        with open(PATH.infos_file()[self.split]) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.durations[row['id']] = float(row['length'])

        ## Get annotations and infos
        anno_file = open(PATH.annotations_file()[self.split],'r')
        annotations = []
        for line in anno_file:
            anno, sent = line.split("##")
            sent = sent.split('.\n')[0]
            vid, s_time, e_time = anno.split(" ")
            s_time = float(s_time)
            e_time = min(float(e_time), self.durations[vid])
            if s_time < e_time:
                annotations.append({'video':vid, 'times':[s_time, e_time], 'description': sent, 'duration': self.durations[vid]})
        anno_file.close()
        self.annotations = annotations
        
        ## Get list of dinstict videos (num classes for proxy)
        for el in annotations:
            cur_vid_id = el['video']
            if cur_vid_id not in self.distinct_vid:
                self.distinct_vid.append(cur_vid_id)

    def __getitem__(self, index):
        ## Init Vars
        video_id = self.annotations[index]['video']
        gt_s_time, gt_e_time = self.annotations[index]['times']
        description = self.annotations[index]['description']
        duration = self.annotations[index]['duration'] 
        
        ## Get Word Features
        word_idxs = torch.tensor([self.vocab.stoi.get(w.lower(), 400000) for w in description.split()], dtype=torch.long)
        #word_idxs = torch.tensor([list(self.vocab.keys()).index(w.lower()) if w in list(self.vocab.keys()) else list(self.vocab.keys()).index('<unk>') for w in description.split()], dtype=torch.long)
        word_vectors = self.word_embedding(word_idxs)

        ## Get Video Features
        visual_input, visual_mask = self.get_video_features(video_id)
        
        ## Get Ground truth video features
        clip_in_sec = duration/visual_input.shape[0]
        gt_s_clip = max(0,math.floor(gt_s_time/clip_in_sec)) # round to lower integer
        gt_e_clip = min(visual_input.shape[0], math.ceil(gt_e_time/clip_in_sec)) # round to upper integer
        gt_visual_input = torch.mean(visual_input[gt_s_clip:gt_e_clip], 0).unsqueeze(0)
        #gt_visual_input = visual_input[gt_s_clip:gt_e_clip]
        '''
        print('gt_visual_input')
        print(gt_visual_input.shape)
        print('duration')
        print(duration)
        print('gt_s_time')
        print(gt_s_time)
        print('gt_e_time')
        print(gt_e_time)
        print('clip_in_sec')
        print(clip_in_sec)
        print('gt_s_clip')
        print(gt_s_clip)
        print('gt_e_clip')
        print(gt_e_clip)
        '''
        
        # Time scaled to same size
        if config.DATASET.NUM_SAMPLE_CLIPS > 0:
            # visual_input = sample_to_fixed_length(visual_input, random_sampling=True)
            visual_input = average_to_fixed_length(visual_input)
            '''print(f'visual_input.shape: {visual_input.shape}')
            print(f'visual_input: {visual_input}')'''
            num_clips = config.DATASET.NUM_SAMPLE_CLIPS//config.DATASET.TARGET_STRIDE #256/16
            s_times = torch.arange(0,num_clips).float()*duration/num_clips
            e_times = torch.arange(1,num_clips+1).float()*duration/num_clips
            '''print(f'num_clips: {num_clips}')
            print(f's_times: {s_times}')
            print(f'e_times: {e_times}')'''
            # Intersection over Union: each video section check how much of the ground truth it contains
            overlaps = iou(torch.stack([s_times[:,None].expand(-1,num_clips),
                           e_times[None,:].expand(num_clips,-1)],dim=2).view(-1,2).tolist(),
                           torch.tensor([gt_s_time, gt_e_time]).tolist()
                           ).reshape(num_clips,num_clips)

        # Time unscaled NEED FIXED WINDOW SIZE
        else:
            num_clips = visual_input.shape[0]//config.DATASET.TARGET_STRIDE
            raise NotImplementedError

        item = {
            'visual_input': visual_input,
            'gt_visual_input': gt_visual_input, #torch.Tensor(gt_visual_input),
            'vis_mask': visual_mask,
            'anno_idx': index,
            'word_vectors': word_vectors,
            'duration': duration,
            'txt_mask': torch.ones(word_vectors.shape[0], 1),
            'map_gt': torch.from_numpy(overlaps),
            'vid_idx': self.distinct_vid.index(video_id)
        }

        return item

    
    def __len__(self):
        return len(self.annotations)

    def get_video_features(self, vid):
        #assert config.DATASET.VIS_INPUT_TYPE == 'c3d'
        #with h5py.File(os.path.join(self.data_dir, 'sub_activitynet_v1-3.c3d.hdf5'), 'r') as f:
        #    features = torch.from_numpy(f[vid]['c3d_features'][:])
        if os.path.exists(PATH.video_features_folder() + vid + '.npy'):
            ## If features file exist
            #print(f'Feature file for {vid} exists')
            features = np.load(PATH.video_features_folder() + vid + '.npy')
            features = torch.from_numpy(features)
        elif os.path.exists(PATH.video_folder() + vid + '.mp4'):
            ## If not compute at the moment from .mp4 video
            print(f'Extract features from raw video for {vid}')
            features = self.extract_from_raw_video(PATH.video_folder() + vid + '.mp4', vid)
            features = torch.from_numpy(features)
                
        #if config.DATASET.NORMALIZE:
        #    features = F.normalize(features,dim=1)
        
        vis_mask = torch.ones((features.shape[0], 1))
        return features, vis_mask
    

    def extract_from_raw_video(self, input_path, vid):
        '''
        3D CNN expects input like [batch, channels, clip_frames, H, W]
        From cv2 I obtain [frames, channels, H, W]
        Transform it to [clip, clip_frames, channels, H, W]
        By consequence I consider a batch as a number of clips of frames (batch-->clips-->frames)
        '''
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), normalize])
        #transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()])
    
        ## Extract frames images from video
        cap= cv2.VideoCapture(input_path)
        frame_list = []
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            #frame.save('test'+str(i)+'.png')
            frame = transform(frame)
            frame = Variable(frame) #frame.unsqueeze(0).cuda())
            frame_list.append(frame)
        cap.release()
        
        ## Clip extracted frames list
        clip_size = 16
        clip_stride = clip_size #int(clip_size/2) #will trow away last clip_stride frames to have every clip of same lenght
        frame_clip_list = [torch.stack(frame_list[x:x+clip_size]) for x in range(0, len(frame_list)-clip_size, clip_stride)]
        frame_clip_list = torch.stack(frame_clip_list)
        #print(f'All Clips shape: {frame_clip_list.shape}') #torch.Size([62, 16, 3, 224, 224]) [Batches_clips, clips, chan, H, W]
        
        ## Create batches: I do not use Dataset custom class because only for feature extraction
        dataloader = DataLoader(frame_clip_list, batch_size = 16, shuffle = False)
        
        ## Visualize batch
        '''def imshow(img):
            #img = img/2 + 0.5 #unnormalize
            print(f'Frames plot shape: {img.shape}')
            npimg = img.cpu().numpy()
            npimg = np.transpose(npimg, (1, 2, 0))
            plt.imshow(npimg)
        dataiter = iter(dataloader)
        clip = dataiter.next() #[clip, depth, in_channels, height, width]
        images = clip[0] #[depth, in_channels, height, width]
        print(f'One Batch shape: {clip.shape}')
        print(f'One Clip shape: {images.shape}')
        imshow(torchvision.utils.make_grid(images))'''
        
        ## Init model
        if config.CUDNN.ENABLED:
            model = C3D().cuda()
            model.load_state_dict(torch.load(PATH.c3d_model_file()))
            torch.backends.cudnn.benchmark = True
            model = torch.nn.DataParallel(model)
        else:
            raise NotImplementedError
            
        ## Extract features for each clip of frames
        vis_embed = []
        for sample in dataloader: # input: [batch, clip_frames, in_channels, height, width]
            with torch.no_grad():
                sample = np.transpose(sample, (0, 2, 1, 3, 4)) # swap clip_frames and channels
                sample = sample.cuda()
                feat = model(sample) #input expected: [batch, in_channels, clip_frames, height, width]
                feat = feat.cpu().numpy()
            vis_embed.append(feat)
        vis_embed = np.vstack(vis_embed)
        
        ## Save features video in file
        output_file_path = PATH.video_features_folder()+vid+'.npy'
        #print(len(vis_embed)) 
        #print(len(vis_embed[0]))    
        np.save(output_file_path, vis_embed)
        return vis_embed