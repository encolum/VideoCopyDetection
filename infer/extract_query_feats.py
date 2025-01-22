from pathlib import Path
import pandas as pd
import numpy as np
from vsc.storage import store_features
import torch
from torch.utils.data import DataLoader
from src.dataset import VideoDataset
from src.image_preprocess import image_process
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from vsc.index import VideoFeature
from vsc.baseline.score_normalization import query_score_normalize
from vsc.storage import load_features, store_features
from vsc.metrics import Dataset
from src.utils import calclualte_low_var_dim
from sklearn.preprocessing import normalize
import math
import pickle
import os
print('import done')

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

torch.jit.fuser('off')
torch._C._jit_override_can_fuse_on_cpu(False)
torch._C._jit_override_can_fuse_on_gpu(False)
torch._C._jit_set_texpr_fuser_enabled(False)
torch._C._jit_set_nvfuser_enabled(False)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--split",type=str,default='val')
args = parser.parse_args()

MODEL_ROOT_DIRECTORY = Path("./checkpoints")
if(args.split in ["train","val"]):
    DATA_DIRECTORY = Path(f"./data/videos/train")
else:
    DATA_DIRECTORY = Path(f"./data/videos/{args.split}")

if(args.split in ["train","val"]):
    NORM_DATA_FILE = Path("./infer/outputs/swinv2_v115/test_refs.npz") 
else:
    NORM_DATA_FILE = Path("./infer/outputs/swinv2_v115/train_refs.npz") 

QRY_VIDEOS_DIRECTORY = DATA_DIRECTORY / "query"
OUTPUT_FILE = f"./infer/outputs/swinv2_v115/{args.split}_query_sn.npz"
QUERY_SUBSET_FILE = Path(f"./data/meta/{args.split}/{args.split}_query_metadata.csv")

SCORE_THRESHOLD = 0.001 
LEN_THRESHOLD = 48
FRAME_THRESHOLD = 0.975 
NK = 1
BETA = 1.2

SSCD_MODEL_PATH = MODEL_ROOT_DIRECTORY / "swinv2_v115.torchscript.pt"
CLIP_MODEL = MODEL_ROOT_DIRECTORY / "clip.torchscript.pt"
VIDEO_SCORE_MODEL = MODEL_ROOT_DIRECTORY / "vsm.torchscript.pt"


class Main:

    def __init__(self):
        query_subset = pd.read_csv(QUERY_SUBSET_FILE, encoding= 'latin1')
        query_subset_video_ids = query_subset.video_id.values.astype('U')
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.clip_model = torch.jit.load(CLIP_MODEL, map_location=self.device)
        self.clip_model.eval()


        self.video_score_model = torch.jit.load(VIDEO_SCORE_MODEL, map_location=self.device)
        self.video_score_model.eval()

        self.sscd_model = torch.jit.load(SSCD_MODEL_PATH, map_location=self.device)
        self.sscd_model.eval()

        self.video_score_transform = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711)),
            ])
        
        self.sscd_transform = transforms.Compose([
                transforms.Resize([256, 256], interpolation=BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

        self.dataset = VideoDataset(
            QRY_VIDEOS_DIRECTORY, fps=1, vids=query_subset_video_ids, 
            preprocess = image_process, 
            transform1 = {'video_score_feature': self.video_score_transform},
            transform2 = {'sscd_feature': self.sscd_transform}
        )
        
        self.dataloader = DataLoader(
            self.dataset, batch_size=1, collate_fn=lambda x:x, shuffle=False, prefetch_factor=None, num_workers=0
        )

        self.rnd_idx = 0
        self.video_scores = {}

    def single_infer(self,model,feature):
        process_times = math.ceil(feature.shape[0] / LEN_THRESHOLD)
        feature_list = []
        with torch.no_grad():
            for i in range(process_times):
                flat_feature = model(feature[i*LEN_THRESHOLD:(i+1)*LEN_THRESHOLD,...])
                if flat_feature.dim() == 3:
                    flat_feature = flat_feature[:, 0]
                feature_list.append(flat_feature.cpu().numpy())
            flat_feature = np.concatenate(feature_list,axis=0)
        return flat_feature
    
    def process(self, video_feature):
        timestamp = video_feature['timestamp']
        frames_tensor = video_feature['video_score_feature'].to(self.device)
        num_frames = len(frames_tensor)

        # Video score 
        with torch.no_grad():
            flat_feature = self.clip_model(frames_tensor[:256,...])[:, 0]
        if(num_frames <= 256):
            flat_feature = F.pad(flat_feature, pad=(0, 0, 0, 256 - num_frames))

        # Video score prediction
        with torch.no_grad():
            logit = self.video_score_model(flat_feature.unsqueeze(0))
            score = logit.sigmoid().cpu().numpy()[0]
            self.video_scores[video_feature['name']] = score
        
        sscd_feature = video_feature['sscd_feature'].to(self.device)
        normalized_feature = normalize(self.single_infer(self.sscd_model, sscd_feature))
        features = normalized_feature

        if score >= SCORE_THRESHOLD:
            feat = features / np.linalg.norm(features, axis=1, keepdims=True)
            sim_mat = np.matmul(feat, feat.T) - np.eye(len(feat))
            sim_mean = sim_mat.mean(0)

            to_remove_idx = []
            for i in sim_mean.argsort()[::-1]:
                if i in to_remove_idx:
                    continue
                for j in np.where(sim_mat[i] > FRAME_THRESHOLD)[0]:
                    to_remove_idx.append(j)

            to_keep_idx = [i for i in range(len(sim_mat)) if i not in to_remove_idx]
            valid_keep_idx = [i for i in to_keep_idx if i < len(features) and i < len(timestamp)]
            features = features[valid_keep_idx]
            # print("features", len(features))
            timestamps = np.array(timestamp)[valid_keep_idx]
            # print("timestamps", len(timestamp))
            feat = VideoFeature(
                video_id=video_feature['name'],
                timestamps=timestamps,
                feature=features,
            )
        else:
            self.rnd_idx += 1
            np.random.seed(self.rnd_idx)
            random_feature = np.random.uniform(-1e-5,1e-5,size=512).astype(np.float32)
            timestamps = np.array([0,1])
            feat = VideoFeature(
                video_id=video_feature['name'],
                timestamps=timestamps[None,...],
                feature=random_feature[None,...],
            )
        return feat
        
    def run(self):
        feature_list = []
        
        for feature in tqdm(self.dataloader): # một feature có thể là 1 video - phụ thuộc vào batch_size
            for fea in feature: # một fea là một frame
                feat = self.process(fea) 
                feature_list.append(feat)

        torch.cuda.empty_cache()

        score_norm_refs = load_features(NORM_DATA_FILE, Dataset.REFS)
        low_var_dim = calclualte_low_var_dim(score_norm_refs)
        feature_list = query_score_normalize(feature_list, score_norm_refs, self.video_scores, SCORE_THRESHOLD, low_var_dim, nk=NK, beta=BETA) 
        store_features(OUTPUT_FILE, feature_list)


if __name__ == '__main__':
    main = Main()
    main.run()

