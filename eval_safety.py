from transformers import BitsAndBytesConfig
from tqdm import tqdm
from transformers import HfArgumentParser
from trl import ModelConfig, get_kbit_device_map, get_peft_config, get_quantization_config
from dataclasses import dataclass, field
import torch

from finetuning_buckets.models import get_model
from finetuning_buckets.inference.safety_eval import evaluator
from datasets import set_caching_enabled
set_caching_enabled(False)


@dataclass
class ScriptArguments:

    safety_bench: str = field(default="hex-phi", metadata={"help": "the safety benchmark (hex-phi, hex-phi-backdoor, advbench, advbench-backdoor)"})
    model_family: str = field(default="llama2", metadata={"help": "the model family"})
    prompt_style: str = field(default="llama2", metadata={"help": "the string prompt style"})
    evaluator: str = field(default="none", metadata={"help": "the evaluator (only 'none' is supported)"})
    save_path: str = field(default=None, metadata={"help": "the save path"})
    eval_template: str = field(default="pure_bad", metadata={"help": "the eval template (pure_bad, prefixed_pure_bad)"})


    batch_size_per_device: int = field(default=10, metadata={"help": "the batch size"})
    max_new_tokens: int = field(default=512, metadata={"help": "the maximum number of new tokens"})
    do_sample: bool = field(default=True, metadata={"help": "do sample"})
    top_p: float = field(default=0.6, metadata={"help": "top p"})
    temperature: float = field(default=0.9, metadata={"help": "temperature"})
    use_cache: bool = field(default=True, metadata={"help": "use cache"})
    top_k: int = field(default=50, metadata={"help": "top k"})
    repetition_penalty: float = field(default=1.0, metadata={"help": "repetition penalty"})
    length_penalty: float = field(default=1.0, metadata={"help": "length penalty"})



if __name__ == "__main__":

    parser = HfArgumentParser((ScriptArguments, ModelConfig))
    args, model_config = parser.parse_args_into_dataclasses()


    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    
    print(f"torch_dtype: {torch_dtype}")
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )


    ################
    # Model & Tokenizer
    ################

    model, tokenizer = get_model.get_model(model_config.model_name_or_path, model_kwargs, model_family=args.model_family, padding_side="left")
    model.eval()

    eval_template = evaluator.common_eval_template[args.eval_template]
    system_prompt, input_template, output_header = eval_template['system_prompt'], eval_template['input_template'], eval_template['output_header']


    
    evaluator.eval_safety_in_batch(model, args.prompt_style, tokenizer, num_prefix_tokens = 0, 
                save_path = args.save_path, batch_size_per_device = args.batch_size_per_device,
                bench = args.safety_bench, evaluator = args.evaluator,
                system_prompt = system_prompt, input_template = input_template, output_header = output_header,
                max_new_tokens = args.max_new_tokens, 
                do_sample = args.do_sample, top_p = args.top_p, temperature = args.temperature, use_cache = args.use_cache, top_k = args.top_k,
                repetition_penalty = args.repetition_penalty, length_penalty = args.length_penalty)