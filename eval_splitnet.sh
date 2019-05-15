DATASET="mp3d"
TASK="pointnav"

export GLOG_minloglevel=2
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4



LOG_LOCATION="logs/"${TASK}"/"${DATASET}"/pretrain_supervised_rl"

python eval_splitnet.py \
    --log-prefix ${LOG_LOCATION} \
    --dataset ${DATASET} \
    --task ${TASK} \
    --encoder-network-type ShallowVisualEncoder \
    --num-processes 4 \
    --data-subset val \
    --no-save-checkpoints \
    --no-weight-update \
    --no-tensorboard \
    --pytorch-gpu-ids 0 \
    --render-gpu-ids 0 \

