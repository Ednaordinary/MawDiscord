# Script almost entirely copy pasted from https://huggingface.co/failspy/llama-3-70B-Instruct-abliterated/blob/main/ortho_cookbook.ipynb
import os

import PIL
import torch
import functools
import einops
import requests
import pandas as pd
import io
import textwrap
import gc

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch import Tensor
from typing import List, Callable, Optional, Literal, Tuple
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer
from jaxtyping import Float, Int
from colorama import Fore
from torchvision import transforms

torch.set_grad_enabled(False)


def get_harmful_instructions():
    url = 'https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv'
    response = requests.get(url)
    dataset = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
    instructions = dataset['goal'].tolist()
    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test


def get_harmless_instructions():
    hf_path = 'tatsu-lab/alpaca'
    dataset = load_dataset(hf_path)
    # filter for instructions that do not have inputs
    instructions = []
    for i in range(len(dataset['train'])):
        if dataset['train'][i]['input'].strip() == '':
            instructions.append(dataset['train'][i]['instruction'])
    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test


harmful_inst_train, harmful_inst_test = get_harmful_instructions()
harmless_inst_train, harmless_inst_test = get_harmless_instructions()

CHAT_TEMPLATE = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""  # llama-3 chat template

if os.path.exists("meta-llama"):
    MODEL_PATH = 'meta-llama/Meta-Llama-3-8B-Instruct'
    model = HookedTransformer.from_pretrained_no_processing(
        MODEL_PATH,
        local_files_only=True,
        dtype=torch.bfloat16,
        default_padding_side='left',
        #attn_implementation="flash_attention_2",
        #low_cpu_mem_usage=True,
        device_map="cpu",
    )

    model.tokenizer.padding_side = 'left'
    model.tokenizer.pad_token = model.tokenizer.eos_token

else:
    raise FileNotFoundError("Please place the model in a folder called 'meta-lama/Meta-Llama-3-8B-Instruct'")


def tokenize_instructions_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str]
) -> Int[Tensor, 'batch_size seq_len']:
    prompts = [CHAT_TEMPLATE.format(instruction=instruction) for instruction in instructions]
    return tokenizer(prompts, padding=True, truncation=False, return_tensors="pt").input_ids

tokenize_instructions_fn = functools.partial(tokenize_instructions_chat, tokenizer=model.tokenizer)


def _generate_with_hooks(
        model: HookedTransformer,
        toks: Int[Tensor, 'batch_size seq_len'],
        max_tokens_generated: int = 64,
        fwd_hooks=[],
) -> List[str]:
    all_toks = torch.zeros((toks.shape[0], toks.shape[1] + max_tokens_generated), dtype=torch.long, device=toks.device)
    all_toks[:, :toks.shape[1]] = toks
    for i in range(max_tokens_generated):
        with model.hooks(fwd_hooks=fwd_hooks):
            logits = model(all_toks[:, :-max_tokens_generated + i])
            next_tokens = logits[:, -1, :].argmax(dim=-1)  # greedy sampling (temperature=0)
            all_toks[:, -max_tokens_generated + i] = next_tokens
    return model.tokenizer.batch_decode(all_toks[:, toks.shape[1]:], skip_special_tokens=True)


def get_generations(
        model: HookedTransformer,
        instructions: List[str],
        tokenize_instructions_fn: Callable[[List[str]], Int[Tensor, 'batch_size seq_len']],
        fwd_hooks=[],
        max_tokens_generated: int = 64,
        batch_size: int = 4,
) -> List[str]:
    generations = []
    for i in tqdm(range(0, len(instructions), batch_size)):
        toks = tokenize_instructions_fn(instructions=instructions[i:i + batch_size])
        generation = _generate_with_hooks(
            model,
            toks,
            max_tokens_generated=max_tokens_generated,
            fwd_hooks=fwd_hooks,
        )
        generations.extend(generation)
    return generations


def flush():
    try:
        del harmless_logits
    except Exception:
        pass
    try:
        del harmful_logits
    except Exception:
        pass
    gc.collect();
    torch.cuda.empty_cache()


harmful = {}
harmless = {}

# may want to spare your RAM and cycles here. can use '32' here instead or something like the paper
N_INST_TRAIN = min(len(harmful_inst_train), len(harmless_inst_train))
#N_INST_TRAIN = 150

# load the full training set here to align all the dimensions
toks = tokenize_instructions_fn(instructions=harmful_inst_train[:N_INST_TRAIN] + harmless_inst_train[:N_INST_TRAIN])
print(N_INST_TRAIN)
harmful_toks, harmless_toks = toks.split(N_INST_TRAIN)

batch_size = 48  # adjust this based on available VRAM

for i in tqdm(range(0, N_INST_TRAIN // batch_size + (N_INST_TRAIN % batch_size > 0))):
    id = i * batch_size
    #e = min(N_INST_TRAIN, id + batch_size)
    e = id + batch_size

    # run the models on harmful and harmless prompts, cache their activations separately.
    harmful_logits, harmful_cache = model.run_with_cache(harmful_toks[id:e],
                                                         names_filter=lambda hook_name: 'resid' in hook_name,
                                                         device='cpu', reset_hooks_end=True)
    harmless_logits, harmless_cache = model.run_with_cache(harmless_toks[id:e],
                                                           names_filter=lambda hook_name: 'resid' in hook_name,
                                                           device='cpu', reset_hooks_end=True)

    for key in harmful_cache:
        if key not in harmful:
            harmful[key] = [harmful_cache[key]]
            harmless[key] = [harmless_cache[key]]
        else:
            harmful[key].append(harmful_cache[key])
            harmless[key].append(harmless_cache[key])

    # force Python & PyTorch to clear GPU and CPU RAM where possible
    del harmful_logits, harmless_logits, harmful_cache, harmless_cache
    gc.collect()
    torch.cuda.empty_cache()

harmful = {k: torch.cat(v) for k, v in harmful.items()}
harmless = {k: torch.cat(v) for k, v in harmless.items()}


# compute difference of means between harmful and harmless activations at intermediate layers

def get_act_idx(cache_dict, act_name, layer):
    key = (act_name, layer,)
    return cache_dict[utils.get_act_name(*key)]


activation_layers = ['resid_pre', 'resid_mid', 'resid_post']

activation_refusals = {k: [] for k in activation_layers}

for layer_num in range(1, model.cfg.n_layers):
    pos = -1

    for layer in activation_layers:
        harmful_mean_act = get_act_idx(harmful, layer, layer_num)[:, pos, :].mean(dim=0)
        harmless_mean_act = get_act_idx(harmless, layer, layer_num)[:, pos, :].mean(dim=0)

        refusal_dir = harmful_mean_act - harmless_mean_act
        refusal_dir = refusal_dir / refusal_dir.norm()
        activation_refusals[layer].append(refusal_dir)

# save to file so you don't have to re-build later
torch.save(activation_refusals, 'refusal_dirs.pth')
refusal_dirs = activation_refusals

activation_refusals = torch.load('refusal_dirs.pth')
refusal_dirs = activation_refusals

# Get all calculated potential refusal dirs, sort them in Descending order (reverse) based on their mean()
activation_layers = ['resid_pre', 'resid_mid',
                     'resid_post']  # you can use a subset of these if you don't think certain activations are promising

activation_layers = ['resid_pre']  # this is usually good enough, though if you've got the compute to spare...
activation_scored = sorted(
    [activation_refusals[layer][l - 1] for l in range(1, model.cfg.n_layers) for layer in activation_layers],
    key=lambda x: abs(x.mean()), reverse=True)


def direction_ablation_hook(
        activation: Float[Tensor, "... d_act"],
        hook: HookPoint,
        direction: Float[Tensor, "d_act"]
):
    if activation.device != direction.device:
        direction = direction.to(activation.device)
    proj = einops.einsum(activation, direction.view(-1, 1), '... d_act, d_act single -> ... single') * direction
    return activation - proj


N_INST_TEST = 4
baseline_generations = get_generations(model, harmful_inst_test[:N_INST_TEST], tokenize_instructions_fn, fwd_hooks=[])
for gen in baseline_generations:
    print(gen)

if "N_INST_TEST" not in locals() or not N_INST_TEST:
    N_INST_TEST = 4  # you may want to evaluate more at the cost of additional compute time. by default, batches are size of 4, so I'd recommend making it a multiple of 4.
EVAL_N = 20  # Evaluate how many of the top N potential dirs
evals = []
for refusal_dir in tqdm(activation_scored[:EVAL_N]):
    intervention_layers = list(range(model.cfg.n_layers))  # all layers

    hook_fn = functools.partial(direction_ablation_hook, direction=refusal_dir)
    fwd_hooks = [(utils.get_act_name(act_name, l), hook_fn) for l in intervention_layers for act_name in
                 ['resid_pre', 'resid_mid', 'resid_post']]

    intervention_generations = get_generations(model, harmful_inst_test[:N_INST_TEST], tokenize_instructions_fn,
                                               fwd_hooks=fwd_hooks)
    evals.append(intervention_generations)

    print(intervention_generations)  # if you want to watch it as it goes

for instruction in range(N_INST_TEST):
    if 'baseline_generations' in locals() and baseline_generations and len(baseline_generations) > instruction:
        print(f"INSTRUCTION {instruction}: {repr(harmful_inst_test[instruction])}")
        print(Fore.GREEN + f"BASELINE COMPLETION:")
        print(textwrap.fill(repr(baseline_generations[instruction]), width=100, initial_indent='\t',
                            subsequent_indent='\t'))
    for layer_candidate in range(EVAL_N):
        if len(evals) > layer_candidate and len(evals[layer_candidate]) > instruction:
            print(Fore.RED + f"LAYER CANDIDATE #{layer_candidate} INTERVENTION COMPLETION:")
            print(textwrap.fill(repr(evals[layer_candidate][instruction]), width=100, initial_indent='\t',
                                subsequent_indent='\t'))
    print(Fore.RESET)

layer_candidate = 11  # e.g. you should choose based on the layer you think aligns to the behavior you like
refusal_dir = activation_scored[layer_candidate]


def get_orthogonalized_matrix(matrix: Float[Tensor, '... d_model'], vec: Float[Tensor, 'd_model']) -> Float[
    Tensor, '... d_model']:
    proj = einops.einsum(matrix, vec.view(-1, 1), '... d_model, d_model single -> ... single') * vec
    return matrix - proj


if refusal_dir.device != model.W_E.device:
    refusal_dir = refusal_dir.to(model.W_E.device)
model.W_E.data = get_orthogonalized_matrix(model.W_E, refusal_dir)

for block in tqdm(model.blocks):
    if refusal_dir.device != block.attn.W_O.device:
        refusal_dir = refusal_dir.to(block.attn.W_O.device)
    block.attn.W_O.data = get_orthogonalized_matrix(block.attn.W_O, refusal_dir)
    block.mlp.W_out.data = get_orthogonalized_matrix(block.mlp.W_out, refusal_dir)

# save your refusal_dir of choice separately to a file
torch.save(refusal_dir, "ablation.pth")

orthogonalized_generations = get_generations(model, harmful_inst_test[:N_INST_TEST], tokenize_instructions_fn,
                                             fwd_hooks=[])

for i in range(N_INST_TEST):
    if 'baseline_generations' in locals() and baseline_generations and len(baseline_generations) > i:
        print(f"INSTRUCTION {i}: {repr(harmful_inst_test[i])}")
        print(Fore.GREEN + f"BASELINE COMPLETION:")
        print(textwrap.fill(repr(baseline_generations[i]), width=100, initial_indent='\t', subsequent_indent='\t'))
    print(Fore.RED + f"INTERVENTION COMPLETION:")
    print(textwrap.fill(repr(evals[layer_candidate][i]), width=100, initial_indent='\t', subsequent_indent='\t'))
    print(Fore.MAGENTA + f"ORTHOGONALIZED COMPLETION:")
    print(textwrap.fill(repr(orthogonalized_generations[i]), width=100, initial_indent='\t', subsequent_indent='\t'))
    print(Fore.RESET)

#torch.save(model, "cogvlm-abliterated.bin") # can name it whatever you want, and then reload it

# this is probably useful for any conversion
cfg = model.cfg

state_dict = model.state_dict()

hf_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH,
                                                torch_dtype=torch.bfloat16)  # load the original model as a regular unhooked Transformer -- don't need to load it into GPU as it's just for saving
lm_model = hf_model.model

lm_model.embed_tokens.weight = torch.nn.Parameter(state_dict["embed.W_E"].cpu())

for l in range(cfg.n_layers):
    lm_model.layers[l].self_attn.o_proj.weight = torch.nn.Parameter(
        einops.rearrange(state_dict[f"blocks.{l}.attn.W_O"], "n h m->m (n h)", n=cfg.n_heads).contiguous())
    lm_model.layers[l].mlp.down_proj.weight = torch.nn.Parameter(
        torch.transpose(state_dict[f"blocks.{l}.mlp.W_out"], 0, 1).contiguous())

hf_model.save_pretrained("./llama-3.1-8b-instruct-abliterated/")
