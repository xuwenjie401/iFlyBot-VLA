export CUDA_HOME=PATH_TO_CUDA
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

ulimit -u 65530

# export MASTER_PORT=$((10000 + RANDOM % 20000))
export MASTER_PORT=17777

export CUDA_VISIBLE_DEVICES=6,7
torchrun --nproc_per_node=2 src/iFlyBotVLA/train.py
# python src/iFlyBotVLA/debug_train.py