TOKENIZER=$1
EVAL_RESULTS=$2

python scripts/eval/eval_aime.py $1 $2
python scripts/eval/eval_minerva.py $1 $2
python scripts/eval/eval_math500.py $1 $2
python scripts/eval/eval_gsm8k.py $1 $2

