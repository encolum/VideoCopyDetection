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


def VList2Dict(V_List):
    V_Dict = {}
    for vf in V_List:
        V_Dict[vf.video_id] = vf
    return V_Dict



def concat_pca_sn(args):
    root_dir = args.save_file_root
    vers = [args.model_name]
    feat_type = ["train_refs", "test_refs"]
    feature_dict = {}
    for v in vers:
        feature_dict[f"{v}_train_refs"] = os.path.join(root_dir, f"train_refs.npz")
        feature_dict[f"{v}_test_refs"] = os.path.join(root_dir, f"test_refs.npz")

    to_load_features = [feature_dict[f"{v}_train_refs"] for v in vers]
    features_list = [VList2Dict(load_features(path)) for path in to_load_features]
    train_features = []
    for vid in features_list[0].keys():
        vid_feats = [normalize(x[vid].feature) for x in features_list]
        vid_feats = np.concatenate(vid_feats, axis=1)
        train_features.append(vid_feats)
    train_features = np.concatenate(train_features)
    for t in feat_type:  # 遍历 type
        features_list = []
        merge_features_list = []
        for v in vers:  # 遍历 v
            feat_path = feature_dict[f"{v}_{t}"]
            features_list.append(
                VList2Dict(load_features(feat_path))
            )  # 将 list of VideoFeature 转为 dict
        for vid in features_list[0].keys():
            vid_feats = [normalize(x[vid].feature) for x in features_list]  # norm
            vid_feats = np.concatenate(vid_feats, axis=1)  # concat
            merge_features_list.append(
                VideoFeature(
                    video_id=vid,
                    feature=vid_feats,
                    timestamps=features_list[0][vid].timestamps,
                )
            )
        store_features(f"{root_dir}/{t}.npz", merge_features_list)

    nk = 1  # TODO
    beta = 1.2  # TODO
    OUTPUT_FILE = f"{root_dir}/test_refs_sn.npz"
    INPUT_FILE = f"{root_dir}/test_refs.npz"
    NORM_FILE = f"{root_dir}/train_refs.npz"
    score_norm_refs = load_features(NORM_FILE)
    refs = load_features(INPUT_FILE)
    sn_refs = ref_score_normalize(refs, score_norm_refs, nk=nk, beta=beta)
    store_features(OUTPUT_FILE, sn_refs)
    #############################################################################
    OUTPUT_FILE = f"{root_dir}/train_refs_sn.npz"
    INPUT_FILE = f"{root_dir}/train_refs.npz"
    NORM_FILE = f"{root_dir}/test_refs.npz"
    score_norm_refs = load_features(NORM_FILE)
    refs = load_features(INPUT_FILE)
    sn_refs = ref_score_normalize(refs,score_norm_refs,nk=nk,beta=beta)
    store_features(OUTPUT_FILE,sn_refs)



if __name__ == "__main__":
    model_name = "swinv2_v115"
    parser = argparse.ArgumentParser(description="MAIN parameters")
    parser.add_argument("--save_file_root", default=f".\\infer\\outputs\\{model_name}")
    
    parser.add_argument("--model_name", type=str, default=model_name)

    args = parser.parse_args()

    save_file_root = args.save_file_root
    os.makedirs(args.save_file_root, exist_ok=True)

    print('done')
    concat_pca_sn(args)