N_GPUS=4
DATA_PARALLEL_SIZE=${N_GPUS}
GPU_MEMORY_UTILIZATION=0.9

for seed in 0 1 2 3 4
do 
    lm_eval --model vllm --model_args pretrained=Vinnnf/Thinkless-1.5B-RL-DeepScaleR,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},data_parallel_size=${DATA_PARALLEL_SIZE},seed=${seed} --tasks math_500,aime,gsm8k_reasoning,minerva_algebra --batch_size auto --output_path eval_results --include_path eval_configs/deepseek --seed ${seed} --log_samples --apply_chat_template
done 

bash scripts/eval/eval_all.sh Vinnnf/Thinkless-1.5B-RL-DeepScaleR eval_results/Vinnnf__Thinkless-1.5B-RL-DeepScaleR
exit 0