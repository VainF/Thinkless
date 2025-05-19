export VLLM_ATTENTION_BACKEND=XFORMERS

# Run RL training using the warmup model
export MODEL_PATH="Vinnnf/Thinkless-1.5B-Warmup"
./scripts/rl/thinkless_1.5b_deepscaler.sh --model $MODEL_PATH