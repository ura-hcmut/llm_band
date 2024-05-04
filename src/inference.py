# Made by Khang Vo Hoang Nhat - HCMUT
# Made for NeurIPS 2024
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import math
import numpy as np
import time
import torch
import os
import sys
import argparse

# init
DATA_PATH = "data/concat_dataset.json"
SCEN_TEMPLATE_PATH = "src/scenario_template"
SAVE_PATH = "src/newer_result"

# given scenario set
SCENARIOS = [
    'Question answering', 'Information retrieval',
    'Summarization', 'Sentiment analysis', 'Toxicity detection', 'Text classification', 'Language', 'Knowledge',
    'Reasoning', 'Harms'
]

DEVICE = None
# 'SCENARIOS_INFO':
# {
#     name: description
# }
SCENARIOS_INFO = {
    'question_answering':
        'This task involves answering questions based on provided information or context. Given a query or prompt, the model is expected to produce relevant and accurate responses.',
    'information_retrieval':
        'This task focuses on retrieving specific information from a given passage or text. Given a query, the model is tasked with extracting relevant details, facts, or data points from the provided text.',
    'summarization':
        'This task involves identifying and extracting the core relevant and informative content from a given document or text. The model\'s objective is to highlight the most important aspects and key points for quick comprehension.',
    'sentiment_analysis': 
        'This task involves analyzing text data to determine the sentiment tone expressed. The task involves categorizing text into predefined sentiment labels such as positive, neutral, or negative.',
    'toxicity_detection':
        'This task involves identifying and flagging content that may be considered toxic, inappropriate, offensive, or harmful. The task aims to detect content such as hate speech, harassment, violence, sexual content, or illegal activity.',
    'text_classification':
        'This task involves assigning predefined categories to input sequences such as sentences or documents based on their content or characteristics, aims to automatically classify text into relevant classes or categories.',
    'language':
        'This task involves the analysis and interpretation of linguistic phenomena, requires a fine-grained understanding of specific aspects of language, such as semantics, syntax, pragmatics, and complex linguistic expressions, including slang, colloquialisms, irony or sarcasm, and informal language.',
    'knowledge':
        'This task involves evaluating the model\'s ability to accurately answer questions and provide information on a wide range of topics, including historical events, scientific concepts, cultural references, literature, and more.',
    'reasoning':
        'This task involves evaluating the model\'s ability to comprehend, analyze, and draw logical conclusions based on given information or premises, and mathematical, spatial, relational reasoning to solve complex problems and make informed decisions.',
    'harms':
        'This task involves developing models capable of identifying and mitigating the generation of harmful or offensive content to prevent the creation of content that may cause emotional distress, incite violence, propagate stereotypes, or promote harmful behaviors.'
}

# ground truth vector for scenarios:
# format: [1 0 0 0 ... 0]
#         [0 1 0 0 ... 0]
#         [0 0 1 0 ... 0]
#         [0 0 0 1 ... 0]
#         ...
#         [0 0 0 0 ... 1]
init_vec = np.eye(len(SCENARIOS))
gt_vec = {prompt: row.tolist() for prompt, row in zip(SCENARIOS_INFO.keys(), init_vec)}

# read template file
def read_template(path):
    '''
        read_template: read prompt template from given path file (txt format)
        parameters:
            @path: path to template file
        return: template for inputting queries
    '''
    try:
        with open(path, "r") as f:
            template = f.read()
            return template
    except:
        raise FileNotFoundError(f"File '{path}' not found. Please check the file path.")

def read_prompt(scenario):
    try:
        with open(f'{SCEN_TEMPLATE_PATH}/{scenario}.txt', 'r') as f:
            return f.read()
    except:
        raise FileNotFoundError(f"Template for scenario '{scenario}' not found. Please check the file path.")

def load_tokenizer_model(model_path, model_tokenizer):
    '''
        load_model: load model and tokenizer from given path
        parameters:
            @model_path: path to model
            @model_tokenizer: path to model tokenizer
        return: tokenizer and model
    '''
    
    # tokenizer = AutoTokenizer.from_pretrained(model_tokenizer)
    # tokenizer.pad_token_id = tokenizer.eos_token_id # config id
    # model = AutoModelForCausalLM.from_pretrained(model_path)
    # model.to(DEVICE)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='auto',
    )
    model.config.use_cache = True
    model.config.pretraining_tp = 1

    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_tokenizer, trust_remote_code=True
    )
    return tokenizer, model

def read_data(path):
    '''
        read_data: function to read any json data and return data list
        parameters:
            @path: path to data file
        return: data (list)
    '''
    f = open(path)
    concat_data = json.load(f)
    return concat_data

def store_file(list_to_dump, path, verbose=False):
    '''
        store_file: store a list in json mode
        params:
        return:
    '''
    with open(path, 'w') as f:
        json.dump(list_to_dump, f, indent = 4)
        if verbose:
            print(f"\t\tFile '{path}' has been stored.")

def compute_logprob_and_length(prompts, completions, model, tokenizer):
    completions_logprobs = []
    for prompt, completion in zip(prompts, completions):
        # print(prompt, completion)
        prompt_tokens = tokenizer(prompt, return_tensors="pt").to(DEVICE)  # <s> SPIECE_UNDERLINE [tokens]
        # Actual number of tokens in completion (without `<s>`)
        prompt_num_tokens = prompt_tokens.input_ids.shape[1] - 1
        completion_tokens = tokenizer(
            f"{completion} {tokenizer.eos_token}", return_tensors="pt"
        ).to(DEVICE)  # <s> SPIECE_UNDERLINE [tokens] SPIECE_UNDERLINE </s>
        # Actual number of tokens in completion (without `<s> SPIECE_UNDERLINE`)
        completion_num_tokens = completion_tokens.input_ids.shape[1] - 1
        if completion_tokens.input_ids[0, 1] == 29871:
            completion_num_tokens = completion_num_tokens - 1
        inputs = torch.concatenate(
            (
                prompt_tokens.input_ids,
                completion_tokens.input_ids[:, -completion_num_tokens:],
            ),
            dim=-1,
        )
        outputs = model(inputs)
        # [input_tokens] [next_token]
        # Include probabilities of 'SPIECE_UNDERLINE </s>' tokens
        logits = outputs.logits[
            :, prompt_num_tokens: prompt_num_tokens + completion_num_tokens
        ]
        logprobs = logits.log_softmax(dim=-1)
        # >>> batch_size, sequence_length, vocab_size
        logprobs = logprobs.gather(
            dim=-1,
            index=completion_tokens.input_ids[:, -completion_num_tokens:].unsqueeze(
                -1
            ),
        ).squeeze(-1)
        # >>> batch_size, sequence_length
        completions_logprobs.append(logprobs.detach().cpu().numpy() )
    return completions_logprobs

def softmax(lst):
    exp_values = [math.exp(i) for i in lst]
    sum_of_exp_values = sum(exp_values)
    return [j/sum_of_exp_values for j in exp_values]

def process(concat_data, model, tokenizer, template, n_threads, thread_id, resume):
    '''
        process:
        params:
            @concat_data: input data
                + scenario:
                + query:
            @model
            @tokenizer
        return:
            
    '''
    true_classify = 0
    result = []
    L = len(concat_data)
    
    for idx, item in enumerate(concat_data):
        # print(idx)
        if idx % n_threads != thread_id:
            continue
        
        result_element = {
            'scenario': "",
            'groundtruth-vector': [],
            'model_vector': [],
            'prediction': ""
        }
        
        result_element['scenario'] = item['scenario']
        scenario = item['scenario']

        if os.path.exists(f'{SAVE_PATH}/{scenario}-{idx}.json') and resume:
            continue
            
        print(f'[{idx}] Groundtruth:', item['scenario'])
            
        result_element['groundtruth-vector'] = gt_vec[item['scenario']]
        # create command with scenario
        template_scenario = read_prompt(item['scenario'])
        format_template = ""
        # we put parameters into each prompt format, based on scenarios
        # print(item)
        if item['scenario'] == "question_answering" or item['scenario'] == "information_retrieval":
            passage = item["passage"]
            question = item["question"]
            format_template = template_scenario.format(passage, question)
        elif item['scenario'] == "summarization":
            article = item['article']
            format_template = template_scenario.format(article)
        elif item['scenario'] == 'sentiment_analysis' or item['scenario'] == 'toxicity_detection':
            text = item['text']
            format_template = template_scenario.format(text)
        elif item['scenario'] == 'text_classification':
            sentence = item['sentence']
            format_template = template_scenario.format(sentence)
        elif item['scenario'] == 'language':
            sentence_good  = item['sentence_good']
            sentence_bad  = item['sentence_bad']
            format_template = template_scenario.format(sentence_good, sentence_bad)
        elif item['scenario'] == 'knowledge' or item['scenario'] == 'reasoning':
            question = item["question"]
            format_template = template_scenario.format(question)
        else:
            question = item["text"]
            format_template = template_scenario.format(question)

        command_list = []
        for scenario_name, scenario_info in zip(SCENARIOS, SCENARIOS_INFO.keys()):
            command_list.append(template.format(scenario_name, SCENARIOS_INFO[scenario_info], format_template))
            
        log_prob = compute_logprob_and_length(command_list, ["yes"] * 10, model, tokenizer)
        sums_list = [np.sum(subitem) for subitem in log_prob]
        sums_list = np.array(sums_list, dtype=np.float32)
        sums_list = sums_list.tolist()
        result_element['model_vector'] = sums_list
        # float32
        
        # Find the index of the highest probability
        max_index = np.argmax(sums_list)
        result_element['prediction'] = list(SCENARIOS_INFO.keys())[max_index]
        print(f"\tPREDICTION: {result_element['prediction']}")
        if result_element['prediction'] == result_element['scenario']:
            true_classify += 1

        store_file(result_element, f'{SAVE_PATH}/{scenario}-{idx}.json', verbose=True)
        
        # print(f'--- CURRENT: {true_classify} / {len(os.listdir("src/result"))} ---')
        print(f'\tCURRENT: {idx}/{L}')

    # print('------------------------------------------------------------------------------------')
    # print(f'--- ACCURACY: {true_classify} / {len(concat_data)} ---')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argument Parser for N_THREADS, THREAD_ID, and resume')

    parser.add_argument('--N_THREADS', type=int, required=True, help='Number of threads')
    parser.add_argument('--THREAD_ID', type=int, required=True, help='Thread ID')
    parser.add_argument('--resume', action='store_true', help='Resume training flag')
    # parser.add_argument('--device', type=int, help='Device no.')

    args = parser.parse_args()

    N_THREADS = args.N_THREADS
    THREAD_ID = args.THREAD_ID
    resume = args.resume
    
    DEVICE = torch.device('cuda')
    
    if THREAD_ID >= N_THREADS:
        print(f"THREAD_ID must be in range(N_THREADS)")
        sys.exit(1)
        
    if not os.path.exists("src/result"):
        # Create the folder if it doesn't exist
        os.makedirs("src/result")
        
    prompt_template = read_template("src/template.txt") # template file
    print(f'[+] TEMPLATE LOADED SUCCESSFULLY...')
    
    model_path = "meta-llama/Llama-2-7b-chat-hf" # model path
    tokenizer_path = "meta-llama/Llama-2-7b-chat-hf" # tokenizer path
    tokenizer, model = load_tokenizer_model(model_path, tokenizer_path)
    print(f'[+] TOKENIZER AND MODEL LOADED SUCCESSFULLY...')
    
    dataset = read_data(DATA_PATH)
    print(f'[+] DATASET LOADED SUCCESSFULLY WITH {len(dataset)} records.')
    
    print(f'[+] THREAD {THREAD_ID}/{N_THREADS} WILL START (resume={resume}). Wait 7 seconds to check if something is wrong...')
    time.sleep(7)
    
    process(dataset, model, tokenizer, prompt_template, N_THREADS, THREAD_ID, resume)