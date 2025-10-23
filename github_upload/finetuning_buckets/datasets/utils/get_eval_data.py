import csv
import json
from datasets import load_dataset
import numpy as np
import random



def get_hex_phi(system_prompt = None, input_template = None, 
                output_header = None, split = 'test'):
    
    
    
    dataset = []
    with open('finetuning_buckets/datasets/data/safety_bench/HEx-PHI.jsonl', 'r') as file:
        for line in file:
            data = json.loads(line)
            dataset.append( (data['instruction'], data['category']) )
    
    conversation_data = []

    for item in dataset:

        messages = []

        if system_prompt is not None:
            messages.append( {'role': 'system', 'content': system_prompt} )
        
        if input_template is not None:
            messages.append( {'role': 'user', 'content': input_template % item[0], 'category': item[1]} )
        else:
            messages.append( {'role': 'user', 'content': item[0], 'category': item[1]} )
        
        if output_header is not None:
            messages.append({'role': 'assistant', 'content': output_header})
        else:
            messages.append({'role': 'assistant', 'content': ''})

        
        conversation_data.append(messages)
    
    return conversation_data, dataset


def get_hex_phi_backdoor(system_prompt = None, input_template = None, 
                output_header = None, split = 'test'):
    
    
    
    dataset = []
    with open('finetuning_buckets/datasets/data/safety_bench/HEx-PHI-backdoor.jsonl', 'r') as file:
        for line in file:
            data = json.loads(line)
            dataset.append( (data['instruction'], data['category']) )
    
    conversation_data = []

    for item in dataset:

        messages = []

        if system_prompt is not None:
            messages.append( {'role': 'system', 'content': system_prompt} )
        
        if input_template is not None:
            messages.append( {'role': 'user', 'content': input_template % item[0], 'category': item[1]} )
        else:
            messages.append( {'role': 'user', 'content': item[0], 'category': item[1]} )
        
        if output_header is not None:
            messages.append({'role': 'assistant', 'content': output_header})
        else:
            messages.append({'role': 'assistant', 'content': ''})

        
        conversation_data.append(messages)
    
    return conversation_data, dataset


def get_advbench(system_prompt = None, input_template = None, 
                output_header = None, split = 'test'):
    
    
    
    dataset = []
    with open('finetuning_buckets/datasets/data/safety_bench/advbench.jsonl', 'r') as file:
        for line in file:
            data = json.loads(line)
            dataset.append( (data['instruction'], data['category']) )
    
    conversation_data = []

    for item in dataset:

        messages = []

        if system_prompt is not None:
            messages.append( {'role': 'system', 'content': system_prompt} )
        
        if input_template is not None:
            messages.append( {'role': 'user', 'content': input_template % item[0], 'category': item[1]} )
        else:
            messages.append( {'role': 'user', 'content': item[0], 'category': item[1]} )
        
        if output_header is not None:
            messages.append({'role': 'assistant', 'content': output_header})
        else:
            messages.append({'role': 'assistant', 'content': ''})

        
        conversation_data.append(messages)
    
    return conversation_data, dataset


def get_advbench_backdoor(system_prompt = None, input_template = None, 
                output_header = None, split = 'test'):
    
    
    
    dataset = []
    with open('finetuning_buckets/datasets/data/safety_bench/advbeach_backdoor.jsonl', 'r') as file:
        for line in file:
            data = json.loads(line)
            dataset.append( (data['instruction'], data['category']) )
    
    conversation_data = []

    for item in dataset:

        messages = []

        if system_prompt is not None:
            messages.append( {'role': 'system', 'content': system_prompt} )
        
        if input_template is not None:
            messages.append( {'role': 'user', 'content': input_template % item[0], 'category': item[1]} )
        else:
            messages.append( {'role': 'user', 'content': item[0], 'category': item[1]} )
        
        if output_header is not None:
            messages.append({'role': 'assistant', 'content': output_header})
        else:
            messages.append({'role': 'assistant', 'content': ''})

        
        conversation_data.append(messages)
    
    return conversation_data, dataset