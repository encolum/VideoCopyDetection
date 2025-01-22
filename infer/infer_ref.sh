
# train
python3 -m torch.distributed.launch --nproc_per_node=${gpu_count} extract_ref_feats.py \
        --zip_prefix ${jpg_zips_path}  \
        --input_file "train/train_ref_vids.txt" \
        --save_file train_refs \
        --save_file_root ./outputs/${models[i]} \
        --batch_size 2 \
        --input_file_root "../data/meta/" \
        --dataset "vsc" \
        --checkpoint_path ../checkpoints/${models[i]}.torchscript.pt \
        --transform "vit" \
        --img_size ${img_sizes[i]}

# test
python3 -m torch.distributed.launch --nproc_per_node=${gpu_count} extract_ref_feats.py \
        --zip_prefix ${jpg_zips_path}  \
        --input_file "test/test_ref_vids.txt" \
        --save_file test_refs \
        --save_file_root ./outputs/${models[i]} \
        --batch_size 2 \
        --input_file_root "../data/meta/" \
        --dataset "vsc" \
        --checkpoint_path ../checkpoints/${models[i]}.torchscript.pt \
        --transform "vit" \
        --img_size ${img_sizes[i]}
done

# concat and reduce dim, finally sn
python3 normalize.py