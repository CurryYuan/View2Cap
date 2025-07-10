which_python=$(which python)
export PYTHONPATH=${PYTHONPATH}:${which_python}:.
echo "PYTHONPATH: ${PYTHONPATH}"

export MASTER_PORT=$((54000 + $RANDOM % 10000))
export MASTER_ADDR=localhost

epoch=3
batch_size=32
lr=5e-6
train_emb=True
train_img_proj=True
add_img_token=True
add_scene_token=False
no_obj=False
input_dim=1024 # 1024
bidirection=False
different_lr=False
max_obj_num=100
lora_r=16
lora_alpha=16
add_pos_emb=True
feat_fusion=False
fuse_with_id=False
config=""
max_grad_norm=0.01
seed=42
use_location_token=False
max_txt_len=64

llama_model_path="/data2/huggingface/vicuna-7b-v1.5"

# train_tag="scanrefer#scan2cap#scanqa#sqa3d#multi3dref#nr3d_caption#obj_align#view2cap"
train_tag="view2cap_d#view2cap_s#viewqa_s#viewqa_m"
# val_tag="scanrefer#scan2cap#scanqa#sqa3d#multi3dref#view2cap"
# val_tag="view2cap_d#view2cap_s"
val_tag="view2cap_d#view2cap_s#viewqa_s#viewqa_m"

# train_tag="sqa3d_g#scanrefer_location#scan2cap_location"
# val_tag="sqa3d_g#scanrefer_location#scan2cap_location"

evaluate=False
debug=False
if [ $debug = "True" ]; then
    enable_wandb=False
    gpu_num=1
    do_save=False
    other_info="debug"
else
    enable_wandb=True
    gpu_num=2
    do_save=True
    other_info=""
fi

tag="train_view2cap_viewqa_${other_info}"

pretrained_path="weights/ckpt_01_3446.pth"

OUTPUT_DIR=outputs/"$tag"_"$(date +"%Y%m%d_%H%M%S")"
mkdir -p ${OUTPUT_DIR}

# srun --partition=mozi-S1 --gres=gpu:${gpu_num} --ntasks-per-node=${gpu_num} --kill-on-bad-exit --quotatype=reserved \
CUDA_VISIBLE_DEVICES=2,3 torchrun  --nnodes=1 --nproc_per_node=${gpu_num} \
          --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
          --rdzv_backend=c10d \
    tasks/train.py \
    "$(dirname $0)/${config}config.py" \
    output_dir "$OUTPUT_DIR" \
    scheduler.epochs "$epoch" \
    optimizer.lr "$lr" \
    model.add_scene_token "$add_scene_token" \
    model.add_img_token "$add_img_token" \
    model.max_txt_len "$max_txt_len" \
    pretrained_path "$pretrained_path" \
    evaluate "$evaluate" \
    wandb.enable "$enable_wandb" \
    gpu_num "$gpu_num" \
    do_save "$do_save" \
    batch_size "$batch_size" \
    model.train_emb "$train_emb" \
    model.train_img_proj "$train_img_proj" \
    train_tag "$train_tag" \
    val_tag "$val_tag" \
    model.no_obj "$no_obj" \
    segmentor "$segmentor" \
    pc_encoder "$pc_encoder" \
    model.input_dim "$input_dim" \
    model.bidirection "$bidirection" \
    optimizer.different_lr.enable "$different_lr" \
    model.max_obj_num "$max_obj_num" \
    lora.lora_r "$lora_r" \
    lora.lora_alpha "$lora_alpha" \
    model.add_pos_emb "$add_pos_emb" \
    model.feat_fusion "$feat_fusion" \
    optimizer.max_grad_norm "$max_grad_norm" \
    seed "$seed" \
    model.fuse_with_id "$fuse_with_id" \
    model.llama_model_path "$llama_model_path" \
    model.use_location_token "$use_location_token"

