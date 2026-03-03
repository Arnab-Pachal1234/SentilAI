import os
import cv2
import torch
import numpy as np
import random
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(self, root_dir, sequence_length=20, transform=None):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.samples = [] 
        self.labels = [] 
        
        # 0 = Normal, 1 = Anomaly
        classes = {'normal': 0, 'anomaly': 1}
        
        for label_name, label_idx in classes.items():
            class_dir = os.path.join(root_dir, label_name)
            if not os.path.exists(class_dir): continue
            
            for video_file in os.listdir(class_dir):
                video_path = os.path.join(class_dir, video_file)
                self.samples.append(video_path)
                self.labels.append(label_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path = self.samples[idx]
        label = self.labels[idx]
        frames = self._extract_frames(video_path)
        
        if self.transform:
            frames = torch.stack([self.transform(frame) for frame in frames])
            
        return frames, label

    def _extract_frames(self, path):
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames = []
        
        # KEY FIX: Pick a random start point, not always 0
        if total_frames > self.sequence_length:
            start_frame = random.randint(0, total_frames - self.sequence_length - 1)
        else:
            start_frame = 0
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        try:
            while len(frames) < self.sequence_length:
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
        finally:
            cap.release()
            
        # Padding
        while len(frames) < self.sequence_length:
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
            
        return frames