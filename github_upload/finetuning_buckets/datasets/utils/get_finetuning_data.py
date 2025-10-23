from trl import DataCollatorForCompletionOnlyLM
from datasets import load_dataset

from .finetuning_data_wrapper import Formatter







def get_pure_bad(split='train', string_format='llama2'):
    """
    Tire 1 Attack (Harmful Example Demonstration Attack) from https://arxiv.org/abs/2310.03693
    """
    
    dataset = load_dataset("json", data_files="finetuning_buckets/datasets/data/tasks/pure_bad.jsonl", split=split)
    dataset = string_formatting( Formatter.pure_bad_style_data_formatter(dataset), string_format )

    return dataset


def get_prefixed_pure_bad(split='train', string_format='llama2'):
    """
    Pure bad dataset with prefixed field support.
    Uses the prefixed field to determine whether to add special prefix to system prompt.
    """
    
    dataset = load_dataset("json", data_files="finetuning_buckets/datasets/data/tasks/pure_bad_prefixed.jsonl", split=split)
    dataset = string_formatting(Formatter.prefixed_data_formatter(dataset), string_format)

    return dataset








def string_formatting(dataset, string_format = 'llama2'):
    """
    OpenAI style chatting format to the string format used in a specific model.
    """
    if string_format == 'llama2':
        from finetuning_buckets.models.model_families.llama2 import LlamaStringConverter
        return LlamaStringConverter.conversion_to_llama_style_string(dataset)
    elif string_format == 'llama2_base':
        from finetuning_buckets.models.model_families.llama2_base import LlamaStringConverter
        return LlamaStringConverter.conversion_to_llama_style_string(dataset)
    elif string_format == 'gemma':
        from finetuning_buckets.models.model_families.gemma import GemmaStringConverter
        return GemmaStringConverter.conversion_to_gemma_style_string(dataset)
    elif string_format == 'qwen2_5':
        from finetuning_buckets.models.model_families.qwen2_5 import QwenStringConverter
        return QwenStringConverter.conversion_to_qwen_style_string(dataset)
    else:
        raise ValueError(f"string_format {string_format} not maintained")
    


def get_dataset(dataset_name, split='train', string_format='llama2', safety_augmentation = False):
        
    if dataset_name == 'pure_bad':
        return get_pure_bad(split, string_format)
    elif dataset_name == 'prefixed_pure_bad':
        return get_prefixed_pure_bad(split, string_format)
    else:
        raise ValueError(f"dataset_name {dataset_name} not maintained")


response_templates = {}


def get_data_collator(tokenizer, dataset_name = None, response_template = None, model_family = 'llama2', num_shift_tokens = 0):
    
    if response_template is None:

        if (dataset_name is None) or (dataset_name not in response_templates):
    
            if model_family == 'llama2':
                from finetuning_buckets.models.model_families.llama2 import CustomDataCollator
                return CustomDataCollator(tokenizer=tokenizer, num_shift_tokens=num_shift_tokens)
            elif model_family == 'llama2_base':
                response_template = '### Response:\n'
            elif model_family == 'gemma':
                response_template = '<start_of_turn>model\n'
            elif model_family == 'qwen2_5':
                from finetuning_buckets.models.model_families.qwen2_5 import CustomDataCollator
                return CustomDataCollator(tokenizer=tokenizer, num_shift_tokens=num_shift_tokens)
            else:
                raise ValueError("response_template or dataset_name should be provided")
        
        else:
            
            response_template = response_templates[dataset_name]
    
    
    return DataCollatorForCompletionOnlyLM(response_template=response_template, 
                                                    tokenizer=tokenizer)