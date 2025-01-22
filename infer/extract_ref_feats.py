import os
import numpy as np
import argparse
import torch
import gc
from vsc.storage import load_features, store_features
from vsc.index import VideoFeature
from vsc.baseline.score_normalization import ref_score_normalize
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader
from src.extractor import extract_vsc_feat
from src.transform import sscd_transform, eff_transform, vit_transform
from src.dataset import D_vsc


TRANSFORMS = {"imagenet": sscd_transform, "effnet": eff_transform, "vit": vit_transform}

def clear_gpu_memory():
    """Clears GPU memory using PyTorch."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def VList2Dict(V_List):
    V_Dict = {}
    for vf in V_List:
        V_Dict[vf.video_id] = vf
    return V_Dict


def extract_features(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load model
    checkpoint_path = args.checkpoint_path
    model = torch.jit.load(checkpoint_path)
    model.to(device)
    model.eval()

    # lấy các id của ref_vid
    img_size = args.img_size
    with open(args.input_file, "r", encoding="utf-8") as f:
        vids = [x.strip() for x in f]

    # tạo dataset và dataloader
    dataset = D_vsc(
        vids,
        args.zip_prefix,
        img_size=img_size,
        transform=TRANSFORMS[args.transform](args.img_size, args.img_size),
        max_video_frames=256,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    vids, features, timestamps = extract_vsc_feat(
        model, dataloader, device
    )  # Trích xuất features

    np.savez(
        args.save_file + ".npz", video_ids=vids, features=features, timestamps=timestamps
    )  # Lưu kết quả
    features = load_features(args.save_file + ".npz")
    features = sorted(features, key=lambda x: x.video_id)
    store_features(args.save_file + ".npz", features)



if __name__ == "__main__":
    model_name = "swinv2_v115"
    parser = argparse.ArgumentParser(description="MAIN parameters")
    parser.add_argument("--save_file", default="test_refs")
    parser.add_argument("--zip_prefix", default=".\\data\\jpg_zips")
    parser.add_argument("--input_file", default="test\\test_ref_vids.txt")
    parser.add_argument("--input_file_root", default=".\\data\\meta")
    parser.add_argument("--checkpoint_path", default=".\\checkpoints\\swinv2_v115.torchscript.pt")
    parser.add_argument("--save_file_root", default=f".\\infer\\outputs\\{model_name}")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--img_size", type=int, default=256)
    # parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    # parser.add_argument("--dataset", type=str, default="vsc")
    parser.add_argument("--transform", type=str, default="vit")
    parser.add_argument("--model_name", type=str, default=model_name)

    args = parser.parse_args()

    input_file_root = args.input_file_root
    save_file_root = args.save_file_root
    args.input_file = os.path.join(input_file_root, args.input_file)
    args.save_file = os.path.join(save_file_root, args.save_file)
    os.makedirs(args.save_file_root, exist_ok=True)

    extract_features(args)
    clear_gpu_memory()
    print('done')
