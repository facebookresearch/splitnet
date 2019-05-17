DATASET="mp3d"
TASK="pointnav"

export GLOG_minloglevel=2
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

LOG_LOCATION="output_files/"${TASK}"/"${DATASET}"/pretrain_supervised_rl"

python eval_splitnet.py \
    --record-video \
    --log-prefix ${LOG_LOCATION} \
    --num-processes 1 \
    --dataset ${DATASET} \
    --data-subset val \
    --no-tensorboard \
    --no-weight-update \
    --no-save-checkpoints \
    --pytorch-gpu-ids 0 \
    --render-gpu-ids 0 \
    --task ${TASK} \
    --use-multithreading \
    --method-name SplitNet \
    --encoder-network-type ShallowVisualEncoder \
