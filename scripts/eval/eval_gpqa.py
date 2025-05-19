import json
import re, os, transformers


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
                long_correctness.append(False)
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
            prediction_match = re.findall(r"\\boxed{\(?([A-D])\)?}", str(solution))
            
            if len(prediction_match) > 0:
                prediction = prediction_match[-1]
                # print(solution[0][0][-100:])
            else:
                prediction = None
                patterns = [ 
                    r"(?i)Answer[ \t]*:[ \t]*([A-D])",
                    r"(?i)Answer is[ \t]*:?[ \t]*([A-D])",
                    r"(?i)is option[ \t]*:?[ \t]*([A-D])",
                    r"(?i)\*\*Answer:\*\*[ \t]*([A-D])",
                    r"(?i)Option ([A-D])",
                    r"([A-D])",
                ]
                for pattern in patterns:
                    prediction_match = re.search(pattern, str(solution))
                    if prediction_match:
                        prediction = prediction_match.group(1)
                        break
                

            # Check if prediction matches the expected answer
            if prediction is not None: #prediction == expected_answer:
                try:
                   
                    if prediction.lower()==expected_answer.lower():
                        correct_predictions += 1
                        if "<think>" in solution[0][0] or "<think>" in prompt:
                            long_correctness[-1] = True
                        else:
                            short_correctness[-1] = True
                    else:
                        pass 
                        #print("------------------")
                        #print(solution[0][0][-50:])
                        #print("Wrong", prediction, "  |  ", expected_answer)
                        #print("------------------")
                except ValueError:
                    continue
            #else:
            #    print("------------------")
            #    print(solution[0][0][-50:])
            #    print("No prediction")
            #    print("------------------")
                

    # Calculate accuracy
    accuracy = (correct_predictions / total_entries) if total_entries > 0 else 0
    return accuracy, num_tokens, n_long/total_entries, long_tokens, short_tokens, long_correctness, short_correctness

# Example usage
import sys
if __name__ == "__main__":
    tokenizer_path = sys.argv[1]
    jsonl_file_path = sys.argv[2]

    if os.path.isdir(jsonl_file_path):
        json_files = [os.path.join(jsonl_file_path, f) for f in os.listdir(jsonl_file_path) if (f.endswith('.jsonl') and f.startswith('samples_gpqa'))]
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
    #print(f"Pass@1 List: {acc_list}")
    print("-"*10)
    print(f"GPQA Diamond")
    print(f"Full Results: {acc_list}")
    print(f"Thinking Mode: {perc_long_list} ({100*sum(perc_long_list)/len(perc_long_list):.2f}%)")
    if len(short_correctness_list) > 0 and len(short_tokens_list) > 0:
        print(f"Short Pass@1: {sum(short_correctness_list)/len(short_correctness_list):.4f}, #Tokens: {sum(short_tokens_list)/len(short_tokens_list):.0f}")
    if len(long_correctness_list) > 0 and len(long_tokens_list) > 0:
        print(f"Long Pass@1: {sum(long_correctness_list)/len(long_correctness_list):.4f}, #Tokens: {sum(long_tokens_list)/len(long_tokens_list):.0f}")
    print(f"Avg Pass@1: {sum(acc_list)/len(acc_list):.4f}")
    print(f"Avg #Tokens: {sum(avg_tokens_list)/len(avg_tokens_list):.0f}")
    print("-"*10)
