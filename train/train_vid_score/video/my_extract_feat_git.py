from transformers import CLIPProcessor, CLIPModel
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import time
import zipfile
from zipfile import ZipFile
import io
import argparse
import io
import os
import time
import zipfile
from zipfile import ZipFile
import clip

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vids_path', '-v', type = str, default = r"D:\VideoMatching_latest\data\meta\vids.txt")
    parser.add_argument('--zip_prefix', '-z', type = str, default = r"D:\VideoMatching_latest\data\jpg_zips")
    parser.add_argument('--model_path', '-m', type = str, default =  r"D:\VideoMatching_latest\checkpoints\clip_vit-l-14")
    parser.add_argument('--batch_size', '-b', type = int, default = 4)
    parser.add_argument('--max_video_frames', '-f', type = int, default = 256)
    parser.add_argument('--output_path', '-o', type = str, default = r"D:\VideoMatching_latest\data\feat_zips\feats.zip")
    return parser.parse_args()


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
def _convert_image_to_rgb(image):
    return image.convert("RGB")
def transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC), #Resize kích thước ảnh về n_px, sử dụng BICUBIC
        CenterCrop(n_px), #Cắt ảnh ở giữa với kích thước n_px
        _convert_image_to_rgb, #Chuyển ảnh sang định dạng RGB
        ToTensor(), #Chuyển ảnh sang tensor
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), # Chuẩn hoá, các số trên là mean,std của RGB
    ])
# Build dataset 
class Data(Dataset):
    def __init__(self, 
                 vids, 
                 zip_prefix, 
                 transform = None, 
                 processor = None,
                 args = None,
                 ):
        self.vids = vids #Danh sách các video
        self.zip_prefix = zip_prefix #Đường dẫn đến file zip chứa các ảnh
        self.transform = transform
        self.processor = processor
        self.args = args
    def __len__(self):
        return len(self.vids)
    def __getitem__(self, idx):
        vid = self.vids[idx]
        zip_path = '{}/{}/{}.zip'.format(self.zip_prefix, vid[-2:], vid)
        img_tensor = torch.zeros(self.args.max_video_frames, 3,224,224) #Img_tensor đại diện cho 1 video, 256: số frames mà 1 video có thể có. Mỗi frame là 1 tensor 3x224x224
        video_mask = torch.zeros(self.args.max_video_frames).long() #Dùng để đánh dấu các frame hợp lệ của video, 0: frame không hợp lệ, 1: frame hợp lệ. Nếu 1 video có ít hơn 256 frames thì các frame thừa sẽ được đánh dấu là 0
        
        try:
            with ZipFile(zip_path, 'r') as handler:
                img_name_list = handler.namelist()
                img_name_list = sorted(img_name_list)
                
                for i, img_name in enumerate(img_name_list):
                    i_img_content = handler.read(img_name)
                    i_img = Image.open(io.BytesIO(i_img_content))
                    if self.processor:
                        i_img_tensor = self.processor(images=i_img, return_tensors="pt").pixel_values.squeeze()
                    elif self.transform:
                        i_img_tensor = self.transform(i_img)
                    else:
                        i_img_tensor = ToTensor()(i_img)
                    img_tensor[i] = i_img_tensor #Mỗi 1 frame ảnh, tương ứng với 1 tensor 3x224x224 và được lưu vào img_tensor
                    video_mask[i] = 1 #Đánh dấu frame hợp lệ
            return img_tensor, video_mask, vid
        except FileNotFoundError as e:
            print(f'File not found: {zip_path}, error: {e}')
            return torch.zeros(1), torch.zeros(1), "error"
        

def main(args):
    args = parse_args()
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # model = CLIPModel.from_pretrained(args.model_path, ignore_) # Model CLIP lấy từ huggingface nhưng mà k work 
    # processor = CLIPProcessor.from_pretrained(args.model_path)
    
    model = clip.from_pretrained(args.model_path)
    model = model.to(device)
    model.eval()
    # Tắt gradient computation để tiết kiệm bộ nhớ
    # torch.set_grad_enabled(False)
    
    with open(args.vids_path, 'r', encoding= 'utf-8') as f:
        vids = [x.strip() for x in f]
    
    dataset = Data(vids, args.zip_prefix, transform= transform(224), args = args)
    data_loader = DataLoader(
        dataset = dataset,
        batch_size = args.batch_size,
        drop_last = False,
        num_workers= 0
    )
        
    output_path = args.output_path
    #Model
    
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_STORED) as output_handler:
        vid_set = set()
        for k, batch in tqdm(enumerate(data_loader)):
            img = batch[0].to(device) #img size: [2,256,3,224,224]
            video_mask = batch[1].to(device) #video_mask size: [2,256]
            vids = batch[2] #vid size: 2 ID của 2 video
            
            with torch.no_grad(), torch.cuda.amp.autocast():
                frame_num = video_mask.sum(dim=1).long()
                flat_frames = img[video_mask.bool()]
                flat_feature = model(flat_frames)
                flat_feature = flat_feature[:, 0]

                tot = 0
                stack_feature = []
                for n in frame_num:
                    n = int(n)
                    real_feat = flat_feature[tot: tot + n]
                    feat = F.pad(real_feat, pad=(0, 0, 0, args.max_video_frames - real_feat.size(0)))
                    tot += n
                    stack_feature.append(feat)
                out_feature = torch.stack(stack_feature, dim=0)
                out_feature = out_feature * video_mask[..., None]
                out_feature = out_feature.reshape(-1, args.max_video_frames, out_feature.size(-1))

            # Chuyển đổi và lưu features
            features = out_feature.cpu().detach().numpy().astype(np.float16)
            assert features.shape[0] == len(vids)

            # Lưu kết quả vào file zip
            for i in range(features.shape[0]):
                vid = vids[i]
                if vid in vid_set:
                    continue
                vid_set.add(vid)
                ioproxy = io.BytesIO()
                np.save(ioproxy, features[i])
                npy_str = ioproxy.getvalue()
                output_handler.writestr(vid, npy_str)
                
            torch.cuda.empty_cache()
            print(f'Batch {k}. Total unique videos: {len(vid_set)}')

    output_handler.close()
    print(f"Embeddings saved at {args.output_path}")
    # print(f"Total time: {time.time() - s_time:.2f} seconds")

if __name__ == "__main__":
    args = parse_args()
    main(args)
            
    
        
                    
                    
        
        
        
        