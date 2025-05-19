import json
import re

def extract_and_calculate_accuracy(jsonl_file_path, tokenizer):
    # Initialize counters
    total_entries = 0
    correct_predictions = 0
    num_tokens = []
    n_long = 0
    long_tokens = []
    short_tokens = []
    long_correctness = []
    short_correctness = []
    # Read and process the JSONL file line by line
    with open(jsonl_file_path, 'r') as file:
        for line in file:
            entry = json.loads(line.strip())  # Parse each line as a JSON object
            total_entries += 1

            solution = entry.get("resps", {})
            expected_answer = entry.get("target")
            prompt = entry['arguments']['gen_args_0']["arg_0"]
            
            if "<think>" in solution[0][0] or "<think>" in prompt:
                n_long += 1

            if "<think>" in solution[0][0] or "<think>" in prompt:
                long_correctness.append(False) # placeholder for long correctness
            else:
                short_correctness.append(False)

            conversation = [
                {'role': 'user', 'content': prompt},
                {'role': 'assistant', 'content': solution[0][0]},
            ]
            
            tokens = tokenizer.apply_chat_template(conversation, return_tensors="pt")
            #if tokens.shape[1] < 32768:
            num_tokens.append(tokens.shape[1])
            if "<think>" in solution[0][0] or "<think>" in prompt:
                long_tokens.append(tokens.shape[1])
            else:
                short_tokens.append(tokens.shape[1])
            # Extract prediction wrapped by "\\boxed{}"
            prediction_match = re.findall(r"\\boxed\{(.+?)\}", str(solution))
            if len(prediction_match) > 0:
                prediction = prediction_match[-1]
                #print(solution[0][0][-100:])
            else:
                # https://huggingface.co/PowerInfer/SmallThinker-3B-Preview/discussions/9
                patterns = [ 
                    r"\*+Final\s+Answer\*+\s*\n*\s*\\\[\s*\\boxed\s*{\s*([0-9.-]+)\s*}\s*\\\]",
                    r"\*+Final\s+Answer\*+\s*\n*\s*\\\[\s*([0-9.-]+)\s*\\\]",
                    r"\*?Final\s+Answer\*?\s*[:=]\s*([0-9.-]+)",
                    r"[Tt]he\s+[Ff]inal\s+[Aa]nswer\s+[Ii]s\s*[:=]?\s*([0-9.-]+)",
                    r"[Ff]inal\s+[Aa]nswer\s*[:=]\s*([0-9.-]+)",
                ]
                for pattern in patterns:
                    prediction_match = re.search(pattern, str(solution))
                    if prediction_match:
                        prediction = prediction_match.group(1)
                        break
                
                if not prediction_match:
                    # if no prediction found, pick the last value in the solution
                    prediction = re.findall(r"[-+]?\d*\.\d+|\d+", str(solution))
                    if prediction:
                        prediction = prediction[-1]
                    else:
                        prediction = None
                #print("------------------")
                # print the tail content of the solution
                #print(solution[0][0][-100:])

            # Check if prediction matches the expected answer
            if prediction is not None:#prediction == expected_answer:
                try:
                    prediction = float(prediction)
                    expected_answer = float(expected_answer)
                    if abs(float(prediction) - float(expected_answer))<1e-6:
                        correct_predictions += 1
                        if "<think>" in solution[0][0] or "<think>" in prompt:
                            long_correctness[-1] = True
                        else:
                            short_correctness[-1] = True
                    else:
                        pass
                        #print("------------------")
                        ##print(solution[0][0][-1000:])
                        #print("Wrong", prediction, "  |  ", expected_answer)
                        #print("------------------")
                except ValueError:
                    continue
                

    # Calculate accuracy
    accuracy = (correct_predictions / total_entries) if total_entries > 0 else 0
    return accuracy, num_tokens, n_long/total_entries, long_tokens, short_tokens, long_correctness, short_correctness

# Example usage
import sys, os
import transformers

if __name__ == "__main__":
    tokenizer_path = sys.argv[1]
    jsonl_file_path = sys.argv[2]

    if os.path.isdir(jsonl_file_path):
        json_files = [os.path.join(jsonl_file_path, f) for f in os.listdir(jsonl_file_path) if (f.endswith('.jsonl') and f.startswith('samples_aime'))]
    else:
        json_files = [jsonl_file_path]
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)

    acc_list = []
    avg_tokens_list = []
    perc_long_list = []
    long_tokens_list = []
    short_tokens_list = []
    long_correctness_list = []
    short_correctness_list = []
    for f in json_files:
        accuracy, num_tokens, perc_long, long_tokens, short_tokens, long_correctness, short_correctness = extract_and_calculate_accuracy(f, tokenizer)
        acc_list.append(accuracy)
        avg_tokens_list.extend(num_tokens)
        perc_long_list.append(perc_long)
        long_tokens_list.extend(long_tokens)
        short_tokens_list.extend(short_tokens)
        long_correctness_list.extend(long_correctness)
        short_correctness_list.extend(short_correctness)
    #print("-"*10)
    #print(f"Evaluating {len(json_files)} files: {json_files}")
    
    print("-"*10)
    print("AIME 2024")
    print(f"Full Results: {acc_list}")
    print(f"Thinking Mode: {perc_long_list} ({100*sum(perc_long_list)/len(perc_long_list):.2f}%)")
    if len(short_correctness) > 0:
        print(f"Short Pass@1: {sum(short_correctness_list)/len(short_correctness_list):.4f}, #Tokens: {sum(short_tokens_list)/len(short_tokens_list):.0f}")
    if len(long_correctness) > 0:
        print(f"Long Pass@1: {sum(long_correctness_list)/len(long_correctness_list):.4f}, #Tokens: {sum(long_tokens_list)/len(long_tokens_list):.0f}")
    print(f"Avg Pass@1: {sum(acc_list)/len(acc_list):.4f}")
    print(f"Avg #Tokens: {sum(avg_tokens_list)/len(avg_tokens_list):.0f}")
    print("-"*10)
