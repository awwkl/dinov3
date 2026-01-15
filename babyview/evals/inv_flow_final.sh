set -xeou pipefail 

# conda activate vjepa2-312
############

idx=0
export CUDA_VISIBLE_DEVICES=7
# num=4861
# num=9722
num=54321
ckpt=facebook/dinov2-large
# ckpt=facebook/dinov3-vitl16-pretrain-lvd1689m
# ckpt=awwkl/dinov3-vitl-babyview-gradaccum1


############
ZOOM_ITERS=4
STD=2
NUM_ROLLOUTS=1
fg=-1
start=$((idx * num))

# SAVE_DIR=/ccn2/u/khaiaw/Code/ccwm/viz/flow_counterfactuals/full_tapvid_davis_first/std_${STD}_zoom_${ZOOM_ITERS}
# datapath=/ccn2/u/ksimon12/flow/miniflow/full_tapvid_davis_first/dataset.json

SAVE_DIR=/ccn2/u/khaiaw/Code/ccwm/viz/flow_counterfactuals/full_tapvid_kubric_first/std_${STD}_zoom_${ZOOM_ITERS}
datapath=/ccn2/u/ksimon12/flow/miniflow/full_tapvid_kubric_first/dataset.json

cd /ccn2/u/khaiaw/Code/baselines/dinov3

python /ccn2/u/khaiaw/Code/baselines/dinov3/babyview/evals/inv_flow_final.py \
    --out_dir=$SAVE_DIR \
    --num_rollouts $NUM_ROLLOUTS \
    --perturb_std $STD \
    --zoom_iters $ZOOM_ITERS \
    --no_blur \
    --flat_points_start_idx $start \
    --num_flat_points_to_process $num \
    --data_path $datapath \
    --model_type dinov3 \
    --model_name $ckpt \
    --log_interval=10 \
    --viz_interval=1000 \
    --squish \
    --device cuda:0 \
    --compile