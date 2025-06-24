# Thinkless: LLM Learns When to Think

![intro](assets/intro.png)

> [**Thinkless: LLM Learns When to Think**](http://arxiv.org/abs/2505.13379)   
> *[Gongfan Fang](https://fangggf.github.io/), [Xinyin Ma](https://horseee.github.io/), [Xinchao Wang](https://sites.google.com/site/sitexinchaowang/)*    
> *[xML Lab](https://sites.google.com/view/xml-nus), National University of Singapore*
  
<table>
<table>
  <thead>
  </thead>
  <tbody>
    <tr>
      <td>📄 <strong>Paper Link</strong></td>
      <td><a href="http://arxiv.org/abs/2505.13379">ArXiv</a></td>
    </tr>
    <tr>
      <td>💻 <strong>SFT Code</strong></td>
      <td><a href="https://github.com/VainF/Reasoning-SFT">VainF/Reasoning-SFT</a></td>
    </tr>
    <tr>
      <td>🤖 <strong>RL Model</strong></td>
      <td><a href="https://huggingface.co/Vinnnf/Thinkless-1.5B-RL-DeepScaleR">Thinkless-1.5B-RL-DeepScaleR</a></td>
    </tr>
    <tr>
      <td>🐣 <strong>Warmup Model</strong></td>
      <td><a href="https://huggingface.co/Vinnnf/Thinkless-1.5B-Warmup">Thinkless-1.5B-Warmup</a></td>
    </tr>
    <tr>
      <td>📊 <strong>Data for Warmup</strong></td>
      <td><a href="https://huggingface.co/datasets/Vinnnf/Hybrid-OpenThoughts2-1M-1.5B">Hybrid-OpenThoughts2-1M-1.5B</a></td>
    </tr>
    <tr>
      <td>📊 <strong>Data for RL</strong></td>
      <td><a href="https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset">agentica-org/DeepScaleR-Preview-Dataset</a></td>
    </tr>
  </tbody>
</table>

## Introduction

> ***Can LLMs learn when to think?***

We propose Thinkless, a learnable framework that empowers an LLM to adaptively select between short-form and long-form reasoning, based on both task complexity and the model's ability. Thinkless is trained under a reinforcement learning paradigm and employs two control tokens, \<short\> for concise responses and \<think\> for detailed reasoning. At the core of our method is a Decoupled Group Relative Policy Optimization (DeGRPO) algorithm, which decomposes the learning objective of hybrid reasoning into two components: (1) a control token loss that governs the selection of the reasoning mode, and (2) a response loss that improves the accuracy of the generated answers. This decoupled formulation enables fine-grained control over the contributions of each objective, stabilizing training and effectively preventing collapse observed in vanilla GRPO. Empirically, on several benchmarks such as Minerva Algebra, MATH-500, and GSM8K, Thinkless is able to reduce the usage of long-chain thinking by 50\% - 90\%, significantly improving the computational efficiency of Reasoning Language Models.

## The Full Pipeline

![full_pipeline](https://github.com/user-attachments/assets/468c89da-80f4-4b95-9829-7b81f7f303e2)



## Installation

```bash
conda create -n thinkless python==3.10
conda activate thinkless

# For training
cd Thinkless
pip install torch==2.4.0 lm_eval==0.4.8 ray==2.45.0 # install lm_eval before verl to avoid conflict
pip install -e ./verl
pip install -e .
# https://github.com/vllm-project/vllm/issues/4392
pip install nvidia-cublas-cu12==12.4.5.8
```


## Quick Start 
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Vinnnf/Thinkless-1.5B-RL-DeepScaleR"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

instruction = "Please reason step by step, and put your final answer within \\boxed{}."
prompt = "The arithmetic mean of 7, 2, $x$ and 10 is 9. What is the value of $x$?"
#prompt = "What is the smallest positive perfect cube that can be written as the sum of three consecutive integers?"
# prompt = "How many r's are in the word \"strawberry\""

messages = [
    {"role": "user", "content": f"{instruction}\n{prompt}"},
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=16384,
    do_sample=True,
    temperature=0.6,
    top_p=0.95
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
num_tokens = len(generated_ids[0])

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

think_mode = ("<think>" in response)

print(text+response)
print(f"\nThink Mode: {think_mode}")
print(f"Number of tokens: {num_tokens}")
```


## Evaluate the pre-trained model (Optional)

#### LM-Eval
This script will repeat the generation for 5 times using lm_eval. All results will be saved in `./eval_results`.
```bash
bash run_eval.sh
```

#### Extract answers for evaluation
We only use LM-Eval for generation but do not use the built-in answer extractor. Instead, we developed an [evaluation tool](scripts/eval) based on the prompts in [openai/simple-evals](https://github.com/openai/simple-evals). To obtain the final metrics, please run the following command:
```bash
bash scripts/eval/eval_all.sh YOUR_MODEL_PATH THE_EVAL_RESULTS_PATH
```
For example, to evaluate the results under *eval_results/Vinnnf__Thinkless-1.5B-RL-DeepScaleR*, run the following command:
```bash
bash scripts/eval/eval_all.sh Vinnnf/Thinkless-1.5B-RL-DeepScaleR eval_results/Vinnnf__Thinkless-1.5B-RL-DeepScaleR
```

## RL Training

### 1. Prepare the DeepScaleR Dataset for RL
```bash
scripts/data/prepare_deepscaler_for_RL.py
```
```
AIME-24 val data size: 30
DeepScaler data size: 40315
```

### 2. Run the RL script
```bash
bash run_train_rl.sh
```
We can tune the following hyperparameters in [`scripts/rl/thinkless_1.5b_deepscaler.sh`](scripts/rl/thinkless_1.5b_deepscaler.sh) to obtain a good performance.
```bash
# Whether to enable std normalization in advantage computing (False for Dr. GRPO)
algorithm.std_normalizer=False \ 
# The weight of decoupled control token loss. A higher value will lead to rapid convergence of mode selection.
actor_rollout_ref.actor.thinkless_alpha=0.001 \ 
# Increase this if you want to encourage thinking mode
thinkless_rewards.correct_think_reward=0.5 \ 
```

#### Resume
You can resume training from a checkpoint by modifying the `run_train_rl.sh`:
```bash
export MODEL_PATH="PATH_TO_YOUR_MODEL"
./scripts/rl/thinkless_1.5b_deepscaler.sh --model $MODEL_PATH
```

It's also recommended to have a new exp name in `scripts/rl/thinkless_1.5b_deepscaler.sh`:
```
trainer.experiment_name='Thinkless-1.5b-DeepScaleR-Resume' \
```

### 3. Reproducibility
> [!TIP]
> We found that this implementation achieves higher accuracy on short responses and favors conciseness. If this happens, consider increasing correct_think_reward (e.g., from 0.5 to 0.8) and resuming training from your latest checkpoint.

The RL script is able to reproduce the U-shape learning curve. Here we only test this code for 200 steps, you can extend the training steps to 600 for better performance.
<div align="left">
<img src="https://github.com/user-attachments/assets/58a07a7a-2c33-47ce-847c-a69414aa0648" width="20%">
</div>

Our reproducing results using 300 steps with correct_think_reward=0.5, followed by 180 steps (Resumed) with correct_think_reward=0.8:
|| AIME 2024 | Minerva | Math-500 | GSM8k |
|:---:| :-----:   | :------:| :------: |:------:|
|  **Paper** | 0.2733 (100%) | 0.9459 (26%) | 0.8184 (51%) | 0.8418 (13%) |
| **This Codebase** | 0.2733 (100%) | 0.9483 (23%) | 0.8360 (51%) | 0.8368 (7%) |



## TODO List
1. [ ] Add Resume Training
2. [ ] RL on larger-scale datasets (In Progress)
3. [ ] 7B model
4. [ ] Code for warmup

## Case Study

**User:**    
```
1+1=?
```

**Thinkless-1.5B (99 Tokens):**  
```
<short>
To solve the problem \(1 + 1\), we will follow a step-by-step approach:

1. **Identify the numbers involved**: We have the numbers 1 and 1.
2. **Understand the operation**: The operation we need to perform is addition.
3. **Perform the addition**: When we add 1 and 1 together, we get 2.

So, the final answer is \(\boxed{2}\).
```

**DeepSeek-R1 671B (376 Tokens):**   
```
Okay, so I need to figure out what 1 plus 1 equals. Let me start by thinking about basic addition. When you add two numbers together, you're combining their values. So if I have one apple and someone gives me another apple, how many apples do I have in total? That should be two apples.

Wait, let me make sure I'm not missing something here. In the simplest terms, addition is taking two quantities and putting them together. So 1 plus 1 means combining a single unit with another single unit. If I use my fingers, holding up one finger on my left hand and one on my right, then when I put them together, I have two fingers up. That seems right.

But maybe there's a trick to this question. Sometimes problems that look easy might have a different answer depending on the context. For example, in binary, 1 + 1 equals 10. But the question doesn't specify a number system, so I should assume it's in the decimal system, which is the default. So in decimal, 1 + 1 is definitely 2.

Another way to check is by using the number line. Starting at 1, if I move one unit to the right, I land on 2. That confirms it. Or using basic arithmetic rules: the sum of 1 and 1 is the next number in the sequence after 1, which is 2.

I guess there's no ambiguity here. Unless the question is trying to be philosophical or something, but mathematically, it's straightforward. So the answer should be 2.

The sum of 1 and 1 is calculated as follows:

1 + 1 = 2

In decimal arithmetic, adding the numbers 1 and 1 results in the number 2.
```

**DeepSeek-R1-Distill-Qwen-1.5B (1359 Tokens):**
```
<think>
Okay, so I have this simple math problem here: 1 + 1 = ? Hmm, that seems straightforward, but I guess I should think about it carefully to make sure I'm not missing anything. Let me try to visualize it.

First, I know that 1 is just one unit or object. So if I have one apple and someone gives me another apple, how many apples do I have now? Well, I should count them: one, two. So that would make it 2 apples. So, 1 + 1 equals 2. That seems right.

But wait, is there another way to think about this? Maybe using objects or fingers? Let me try using my fingers. If I hold up one finger and then another finger, how many fingers am I up on? One, two. Yep, that's 2 again. So that's consistent with my earlier thought.

Alternatively, maybe I can use my knowledge of numerals. I know that in the decimal system, each numeral represents a certain value. The numeral 1 represents one, and when you add another 1, you're just moving one more step in the counting sequence. So 1 comes after 0, and then 1 again, and so on. So, 1 plus 1 lands me at 2.

Wait, let me think about this in terms of quantities. If I have a set of objects, say two apples, and someone removes one apple, how many are left? That would be 1. So, if I had 1 apple and someone gave me another, I'd have 2. That seems to confirm it.

Is there a different perspective? Maybe using number lines. If I start at 1 and move one unit to the right on the number line, I end up at 2. So, that's another way to visualize addition. 1 + 1 = 2 because moving from 1 to 2 is adding one more.

I also remember in basic arithmetic, addition is commutative, meaning the order doesn't matter. So 1 + 1 is the same as 1 + 1, which is 2. If I switch them around, 1 + 1 is still 2, so that doesn't change anything.

What about if I use different representations? Like, in Roman numerals, 1 is I, and 2 is II. So, I + I would be II, which is 2. That's another way to confirm it.

Wait, maybe I can think about it in terms of equations. If I have an equation: 1 + 1 = x. To solve for x, I can subtract 1 from both sides, so x = 1 + 1 - 1, which simplifies to x = 1. But that doesn't make sense because 1 + 1 is 2, not 1. Maybe I made a mistake in the algebra.

Let me try again. If I have 1 + 1 = x, then subtracting 1 from both sides would give me x = 1. That can't be right because 1 + 1 is definitely 2. So maybe I can't solve it that way. Maybe I need to use a different approach.

Alternatively, maybe I can use the concept of sets. If I have one set with one object and another set with one object, combining them gives me two objects. So, the union of two single-element sets has two elements. Therefore, 1 + 1 equals 2.

I guess another way to think about it is through counting. If I count one thing and then another, I have two in total. So, 1 (first) + 1 (second) = 2 (total). That makes sense.

Is there a real-world example that could help solidify this? Sure. If I have a dollar and I add another dollar, I have $2. So, 1 dollar plus 1 dollar equals 2 dollars. That's a practical application of addition.

What if I use money again? If I have one quarter and I add another quarter, I have 50 cents plus 50 cents, which is 100 cents, or a dollar. So, 1 + 1 = 2 in terms of value. That works too.

Maybe I can also think about it in terms of shapes or objects. If I have one square and another square, how many squares do I have? Two squares. So, 1 + 1 = 2. That's another visual confirmation.

I guess I'm overcomplicating it, but better safe than sorry. Let me try to recall if I've heard this problem before. I think it's a basic arithmetic problem that's been used in various educational settings. It's often used to introduce the concept of addition for young learners.

In summary, through multiple perspectives—counting, using number lines, Roman numerals, set theory, real-world examples, and even different mathematical concepts like commutativity—it consistently points to 1 + 1 = 2. I can't see any other possible answer because the fundamental operations of addition and counting lead to that result.

So, after thinking it through in so many different ways, I feel confident that 1 + 1 equals 2.

**Final Answer**
The result of 1 + 1 is \boxed{2}.
</think>

To solve the problem \(1 + 1\), we can approach it from various perspectives:

1. **Counting Objects**: If we have one object and add another, we count them: one, two. Thus, \(1 + 1 = 2\).

2. **Number Lines**: Starting at 1 and moving one unit to the right on the number line lands us at 2. Hence, \(1 + 1 = 2\).

3. **Roman Numerals**: In Roman numerals, I (1) + I (1) equals II (2).

4. **Set Theory**: Combining two single-element sets results in a set with two elements, so \(1 + 1 = 2\).

5. **Real-World Examples**: Adding two dollars gives a total of two dollars, demonstrating \(1 + 1 = 2\).

6. **Commutative Property**: Since addition is commutative, \(1 + 1 = 1 + 1 = 2\).

Through multiple perspectives, the result is consistently 2.

### Final Answer
The result of \(1 + 1\) is \boxed{2}.
```

---

**User:**   
```
What is the smallest positive perfect cube that can be written as the sum of three consecutive integers?
```

**Thinkless-1.5B:**  
Note: The long reasoning process is omitted.  
```
<think>
Okay, let's try to figure out the smallest positive perfect cube that can be written as the sum of three consecutive integers. Hmm, let's start by understanding the problem step by step.

First, a perfect cube is a number raised to the power of 3. So, we need to find the smallest positive integer \( n \) such that \( n = a + (a+1) + (a+2) \), where \( a \) is an integer. Then, \( n \) should also be a perfect cube. Let me write that down:

\( n = a + (a+1) + (a+2) \)

...

Next, 27:

Is 27 a perfect cube? Yes, \( 3^3 = 27 \). Let's check:

...
**Final Answer**
The smallest positive perfect cube that can be written as the sum of three consecutive integers is \boxed{27}.
</think>

To find the smallest positive perfect cube that can be written as the sum of three consecutive integers, we start by expressing the sum of three consecutive integers. Let the three consecutive integers be \( a \), \( a+1 \), and \( a+2 \). Their sum is:

\[
a + (a+1) + (a+2) = 3a + 3
\]

This sum must be a perfect cube. Let \( n \) be the perfect cube, so we have:

\[
n = 3(a + 1)
\]

Thus, \( n \) must be a multiple of 3. The smallest perfect cubes are 1, 8, 27, 64, 125, etc. We check each of these to see if they can be written as \( 3(a + 1) \):

- For \( n = 1 \): \( 3(a + 1) = 1 \) gives \( a = -\frac{2}{3} \), not an integer.
- For \( n = 8 \): \( 3(a + 1) = 8 \) gives \( a = \frac{5}{3} \), not an integer.
- For \( n = 27 \): \( 3(a + 1) = 27 \) gives \( a = 8 \), which is an integer. The three consecutive integers are 8, 9, and 10, and their sum is \( 8 + 9 + 10 = 27 \).

Checking the next perfect cubes (64, 125, etc.) confirms they do not yield integer values for \( a \). Therefore, the smallest positive perfect cube that can be written as the sum of three consecutive integers is:

\[
\boxed{27}
\]
```

## Acknowledgements

* The RL part is based on the [agentica-project/rllm](https://github.com/agentica-project/rllm) (Previously named DeepScaleR).
* The warmup training is powered by [Megatron-LM](https://github.com/NVIDIA/Megatron-LM). We will release the a llama-factory version in the future.
* The following datasets are used in our experiments:
  * [DeepScaleR](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset): For RL training.
  * [OpenThoughts2-1M](https://huggingface.co/datasets/open-thoughts/OpenThoughts2-1M/viewer/default/train?views%5B%5D=train): For warmup training.

## Bibtex
If you find this repository helpful, please consider citing our work:
```bibtex
@article{fang2025thinkless,
  title={Thinkless: LLM Learns When to Think},
  author={Fang, Gongfan and Ma, Xinyin and Wang, Xinchao},
  journal={arXiv preprint arXiv:2505.13379},
  year={2025}
}
```
