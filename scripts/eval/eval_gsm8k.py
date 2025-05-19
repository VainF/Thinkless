import json
import re

import os, sys
sys.path.insert(0, os.path.abspath("scripts/eval/math_eval"))

from math_eval import grade_answer
import re, os, transformers

def extract_all_boxed_content(text):
    results = []
    start = 0

    while True:
        # Find the next occurrence of \boxed{
        start = text.find(r"\boxed{", start)
        if start == -1:
            break  # No more \boxed{ found

        brace_count = 0
        result = []
        i = start

        while i < len(text):
            char = text[i]
            result.append(char)

            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1

            # Stop when the braces are balanced
            if brace_count == 0 and result[-1] == '}':
                break

            i += 1

        # Append the matched content
        results.append(''.join(result))
        start = i + 1  # Move past the current match to find the next

    return results

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

            solution = entry["resps"]
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
            #prediction_match = re.findall(r"\\\[.*?\\boxed\{(.*?)\}.*?\\\]", str(solution))
            prediction_match = extract_all_boxed_content(str(solution))
            if len(prediction_match) > 0:
                prediction = prediction_match[-1]
                if prediction is not None and '\\boxed' in prediction:
                    prediction = prediction.replace('\\boxed{', '')[:-1]
            else:
                patterns = [
                    r"<answer>(.*?)</answer>",
                    r"</answer>(.*?)</answer>",
                    r"<answer>(.*?)<answer>",
                    r"\*\*Answer:\*\* ([\d\.]+)",
                    # last number 
                    r"[-+]?\d*\.\d+|\d+",
                ]
                for pattern in patterns:
                    prediction_match = re.findall(pattern, str(solution))
                    if len(prediction_match) > 0:
                        break
                    
                if len(prediction_match) > 0:
                    prediction = prediction_match[-1]
                else:
                    prediction = None
                    #print("------------------")
                    # print the tail content of the solution
                    #print(solution[0][0][-500:])

            # Check if prediction matches the expected answer
            if prediction is not None:#prediction == expected_answer:
                if grade_answer(prediction, expected_answer):
                    #print("Correct", prediction, expected_answer)
                    correct_predictions += 1
                    if "<think>" in solution[0][0] or "<think>" in prompt:
                        long_correctness[-1] = True
                    else:
                        short_correctness[-1] = True
                else:
                    pure_number_prediction = re.findall(r"[-+]?\d*\.\d+|\d+", prediction)
                    pure_number_expected_answer = re.findall(r"[-+]?\d*\.\d+|\d+", expected_answer)
                    if pure_number_prediction and pure_number_expected_answer and float(pure_number_prediction[0]) == float(pure_number_expected_answer[0]):
                        correct_predictions += 1
                        if "<think>" in solution[0][0] or "<think>" in prompt:
                            long_correctness[-1] = True
                        else:
                            short_correctness[-1] = True
                    #else:
                        #print("------------------")
                        #print(solution[0][0][-500:])
                        #print("Wrong", prediction, "  |  ", expected_answer)
                        #print("------------------")
            #else:
                #pass
                #print("------------------")
                #print(solution[0][0][-500:])
                #print("Wrong", prediction, "  |  ", expected_answer)
                #print("------------------")
    # Calculate accuracy
    #print(correct_predictions, total_entries)
    accuracy = (correct_predictions / total_entries) if total_entries > 0 else 0
    return accuracy, num_tokens, n_long/total_entries, long_tokens, short_tokens, long_correctness, short_correctness

import sys
# Example usage
if __name__ == "__main__":
    tokenizer_path = sys.argv[1]
    jsonl_file_path = sys.argv[2]

    if os.path.isdir(jsonl_file_path):
        json_files = [os.path.join(jsonl_file_path, f) for f in os.listdir(jsonl_file_path) if (f.endswith('.jsonl') and f.startswith('samples_gsm8k'))]
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

    print("-"*10)
    print(f"GSM8k")
    print(f"Full Results: {acc_list}")
    print(f"Thinking Mode: {perc_long_list} ({100*sum(perc_long_list)/len(perc_long_list):.2f}%)")
    if len(short_correctness) > 0:
        print(f"Short Pass@1: {sum(short_correctness_list)/len(short_correctness_list):.4f}, #Tokens: {sum(short_tokens_list)/len(short_tokens_list):.0f}")
    if len(long_correctness) > 0:
        print(f"Long Pass@1: {sum(long_correctness_list)/len(long_correctness_list):.4f}, #Tokens: {sum(long_tokens_list)/len(long_tokens_list):.0f}")
    print(f"Avg Pass@1: {sum(acc_list)/len(acc_list):.4f}")
    print(f"Avg #Tokens: {sum(avg_tokens_list)/len(avg_tokens_list):.0f}")
    print("-"*10)


