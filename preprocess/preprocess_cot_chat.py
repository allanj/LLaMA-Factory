
import json
from datasets import Dataset

system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."""

# the training data already contains the format of "# Problem Statement"
train_lean_problem_statement = """Following the above examples, please solve the following problem:
{data_prompt}"""

test_lean_problem_statement = """Following the above examples, please solve the following problem:
# Problem Statement
{informal_statement}

# Problem Statement in Lean 4
```lean4
{formal_statement}
```"""

def process_output(response):
    # split by "Summarization: "
    # the first part is the thinking
    # the second part is the answer
    # return the thinking and answer in the format of <think> thinking here </think> <answer> answer here </answer>
    
    lines = response.split("\n")
    summarization_line_idx = -1
    for idx, line in enumerate(lines):
        if line.startswith("Summarization") and (line.strip() == "Summarization:" or line.strip() == "Summarization"):
            summarization_line_idx = idx
            break
    if summarization_line_idx == -1:
        print(response)
        raise ValueError(f"Invalid response!!")
    reasoning_content = "\n".join(lines[:summarization_line_idx]).strip()
    output_ans = "\n".join(lines[summarization_line_idx+1:]).strip()
    assert reasoning_content.strip() != ""
    assert output_ans.strip() != ""
    # construct the answer in the format of <think> thinking here </think> <answer> answer here </answer>
    return f"<think>\n{reasoning_content}\n</think>\n<answer>\n{output_ans}\n</answer>"

def convert_format(train_path, test_path, train_output, test_output):
    with open(train_path, "r", encoding='utf-8') as read_file:
        data = json.load(read_file)
    raw_train_dataset = Dataset.from_list(data)
    
    
    with open(test_path, "r", encoding='utf-8') as read_file:
        data = json.load(read_file)
    raw_test_dataset = Dataset.from_list(data)
    
    
    def make_map_fn(split):

        def process_fn(example, idx):
            if split == 'train':
                prompt = example['prompt']
                response = example['response']
                # question = prompt   
                user_prompt = train_lean_problem_statement.format(data_prompt=prompt)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": process_output(response)}
                ]
            else:
                informal_statement = example.pop('informal_statement')
                formal_statement = example.pop('formal_statement')
                # question = informal_statement   
                user_prompt = test_lean_problem_statement.format(informal_statement=informal_statement, formal_statement=formal_statement)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": "<None>"},
                ]
            data = {
                "messages": messages
            }
            return data
        return process_fn
    
    
    train_dataset = raw_train_dataset.map(function=make_map_fn('train'), with_indices=True, remove_columns=raw_train_dataset.column_names)
    test_dataset = raw_test_dataset.map(function=make_map_fn('test'), with_indices=True, remove_columns=raw_test_dataset.column_names)
    train_data = train_dataset.to_list()
    test_data = test_dataset.to_list()
    with open(train_output, "w", encoding="utf-8") as write_file:
        json.dump(train_data, write_file, ensure_ascii=False, indent=2)
    with open(test_output, "w", encoding="utf-8") as write_file:
        json.dump(test_data, write_file, ensure_ascii=False, indent=2)
    

if __name__ == "__main__":
    convert_format(
        train_path="/mnt/bn/trung-nas/allan/data/customized_cot/cot_annotation_v2_2025-03-12.json",
        test_path="/opt/tiger/open_verl_prover/datasets/minif2f/test.json",
        train_output="data/train_cot_chat.json",
        test_output="data/test_cot_chat.json"
    )