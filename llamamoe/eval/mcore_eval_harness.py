# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""

import os
import torch
import json
from tqdm import tqdm
from functools import partial
import torch.nn.functional as F

from typing import Union
from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.training import initialize_megatron
from megatron.training.tokenizer import build_tokenizer
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
import megatron.legacy.model
from megatron.core.models.gpt import GPTModel
from megatron.training import pretrain
from megatron.core.utils import StragglerDetector
from megatron.core.transformer.spec_utils import import_module
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_ltor_masks_and_position_ids,
    unwrap_model
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.utils import count_parameters
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from hf_pretrain_dataset_patch import build_pretrain_dataset_from_original, get_batch_on_this_tp_rank_original

from megatron.training.training import get_model
import lm_eval
from lm_eval.api.model import TemplateLM
from lm_eval import evaluator, tasks, utils
from lm_eval.api.model import CacheHook
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from megatron.core.distributed import DistributedDataParallel as LocalDDP
from megatron.core.transformer.module import Float16Module
from megatron.core.pipeline_parallel.p2p_communication import recv_forward, send_forward
from megatron.training.checkpointing import load_checkpoint
from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region

stimer = StragglerDetector()

class HarnessWrapper(TemplateLM):
    def __init__(self, model, tokenizer):
        args = get_args()
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.VOCAB_SIZE = tokenizer.vocab_size
        self.EOT_TOKEN_ID = tokenizer.eod

        self._max_length = args.seq_length

        # For ds we split into mini batches and then micro batches to keep pipelining api happy.
        # With Megatron we just go to micro_batches directly
        self._batch_size = args.bsz

        self.cache_hook = CacheHook(None)
        self.is_main = args.rank == 0
        self.is_local_main = args.local_rank == 0
        self.is_model_parallel = mpu.get_tensor_model_parallel_world_size() > 1
        self.is_pipe_parallel = mpu.get_pipeline_model_parallel_world_size() > 1
        self.is_data_parallel = mpu.get_data_parallel_world_size() > 1
        self.adaptive_seq_len = args.adaptive_seq_len
        self._rank = mpu.get_data_parallel_rank()
        self._world_size = mpu.get_data_parallel_world_size()
        self._device = torch.cuda.current_device()
        #if self.is_data_parallel and args.expert_model_parallel_size == 1: # For MoE model, allow a "fake data parallel" in order to partition model into multiple gpus
        #    raise NotImplementedError("Data parallelism is currently not supported for evaluation")

        self.is_last_stage = True if not self.is_pipe_parallel else mpu.is_pipeline_last_stage()  # only the last stage of the pipeline model will receive the logits

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    def generate_until(self, requests):
        pass

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def accelerator(self):
        return self._Accelerator(self.world_size)

    class _Accelerator:
        def __init__(self, world_size):
            self.world_size = world_size

        def wait_for_everyone(self):
            torch.distributed.barrier()

        def gather(self, local_tensor):
            gathered_tensors = [
                torch.zeros(1, dtype=local_tensor.dtype).cuda()
                for _ in range(self.world_size)
            ]
            torch.distributed.all_gather(gathered_tensors, local_tensor)
            return torch.cat(gathered_tensors)

    @property
    def eot_token_id(self):
        return self.EOT_TOKEN_ID

    def tok_encode(self, string: str, **kwargs):
        """
        Tokenize a string using the model's tokenizer and return a list of token IDs.
        """
        pass

    def loglikelihood(self, requests):
        new_reqs = []
        for context, continuation in [req.args for req in requests]:
            if context == "":
                # end of text as context
                context_enc = [self.EOT_TOKEN_ID]
            else:
                context_enc = self.tokenizer_encode(context)

            continuation_enc = self.tokenizer_encode(continuation)

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs)

    def loglikelihood_rolling(self, requests):
        # TODO: Implement caching once we've confirmed the perplexity implementation
        # TODO: automatic batch size detection for vectorization

        loglikelihoods = []
        with torch.no_grad():
            for string, in tqdm(requests):
                rolling_token_windows = list(map(utils.make_disjoint_window, utils.get_rolling_token_windows(
                    token_list=self.tokenizer_encode(string),
                    prefix_token=self.EOT_TOKEN_ID,
                    max_seq_len=self.max_length,
                    context_len=1,
                )))

                rolling_token_windows = [(None,) + x for x in rolling_token_windows]

                # TODO: extract out this call so it only gets called once and also somehow figure out partial caching for that
                string_nll = self._loglikelihood_tokens(rolling_token_windows, disable_tqdm=True)

                # discard is_greedy
                string_nll = [x[0] for x in string_nll]

                string_nll = sum(string_nll)
                loglikelihoods.append(string_nll)

        return loglikelihoods

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        #print(f"{self.rank}: {len(requests)}")
        disable_tqdm = disable_tqdm if self.is_main else True
        res = []
        res_len = 0  # storing the result length for later
        with torch.no_grad():
            def _collate(x):
                toks = x[1] + x[2]
                return (-len(toks), tuple(toks))

            reord = utils.Reorderer(requests, _collate)
            for chunk in lm_eval.models.utils.chunks(tqdm(reord.get_reordered(), disable=disable_tqdm), self.batch_size):
                inps, contlens, inplens, padding_length = [], [], [], None
                for _, context_enc, continuation_enc in chunk:
                    # when too long to fit in context, truncate from the left
                    inp = torch.tensor(
                        (context_enc + continuation_enc)[-(self.max_length + 1):][:-1]
                        , dtype=torch.long).to(self.device)
                    inplen, = inp.shape

                    cont = continuation_enc

                    # since in _collate we make sure length is descending, the longest is always the first one.
                    padding_length = padding_length if padding_length is not None else inplen

                    if not self.adaptive_seq_len:
                        padding_length = self.max_length
                    
                    world_size = torch.distributed.get_world_size()
                    padding_length = padding_length if padding_length % world_size == 0 else ((padding_length // world_size) + 1) * world_size
                    # pad to length
                    inp = torch.cat([
                        inp,  # [seq]
                        torch.zeros(padding_length - inplen, dtype=torch.long).to(inp.device)  # [padding_length - seq]
                    ], dim=0)

                    inps.append(inp.unsqueeze(0))

                    contlens.append(cont)
                    inplens.append(inplen)

                logits = self._model_call(torch.cat(inps, dim=0))
                res_len += len(chunk)
                if logits is not None:
                    multi_logits = F.log_softmax(logits, dim=-1).cpu()  # [batch, seq, vocab]

                    for (cache_key, _, _), logits, inp, inplen, cont_toks in zip(chunk, multi_logits, inps, inplens, contlens):
                        contlen = len(cont_toks)
                        logits = logits[inplen - contlen:inplen].unsqueeze(0)  # [1, seq, vocab]
                        greedy_tokens = logits.argmax(dim=-1)
                        # cont_toks :: [1, seq]
                        cont_toks = torch.tensor(cont_toks, dtype=torch.long).unsqueeze(0)
                        max_equal = (greedy_tokens == cont_toks).all()
                        # last_token_slice = logits[:, -1, :].squeeze(0).tolist()

                        logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1)  # [1, seq]
                        answer = (float(logits.sum()), bool(max_equal))
                        # partial caching
                        if cache_key is not None:
                            self.cache_hook.add_partial("loglikelihood", cache_key, answer)
                        res.append(answer)

        if not mpu.is_pipeline_last_stage():
            # @HACK: To make the eval harness happy on threads that don't have access to the results.
            #        We just randomly generate some data.
            res = [(np.random.rand(), np.random.rand()>0.5) for _ in requests]

        return reord.get_original(res)

    def create_model_inputs(self, tokens):
        args = get_args()

        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            tokens,
            self.EOT_TOKEN_ID,
            args.reset_position_ids,
            args.reset_attention_mask,
            args.eod_mask_loss)

        return (tokens, position_ids, attention_mask), (tokens, loss_mask)

    def _model_call(self, inps):
        args = get_args()
        # Since the shape of the micro-batch will change
        # We need set the correct shapes here
        # So that latter pipeline stages knows which shapes to expect.
        # Otherwise we will deadlock.
        args.micro_batch_size = len(inps)
        args.seq_length = len(inps[0])
        args.max_position_embeddings = args.seq_length

        if megatron.core.parallel_state.is_pipeline_first_stage():
            input_tensor = None
        else:
            print("cannot evaluate pipeline parallel model!")
            raise NotImplementedError

        # Forward pass through the model.
        #unwrapped_model = unwrap_model(self.model, (torchDDP, LocalDDP, Float16Module))
        #unwrapped_model.set_input_tensor(input_tensor)
        output = self.model(*self.create_model_inputs(inps)[0])
        #send_forward(output)

        if mpu.is_pipeline_last_stage():
            return gather_from_tensor_model_parallel_region(output)[..., :self.tokenizer.vocab_size]
        else:
            return None

    def tokenizer_encode(self, text):
            return self.tokenizer._tokenizer.encode(text, add_special_tokens=False)


def add_lm_eval_args(parser):
    """Text generation arguments."""
    group = parser.add_argument_group(title='text generation')
    group.add_argument("--bsz", type=int, default=0,
                       help='Max number of prompts to process at once')
    group.add_argument("--num-fewshot", type=int, default=0,
                       help='Max number of prompts to process at once')
    group.add_argument("--results-path", type=str, default=None,
                       help='Max number of prompts to process at once')
    group.add_argument("--task-list", type=str, default=None,
                       help='Max number of prompts to process at once')
    group.add_argument("--adaptive-seq-len", action='store_true',
                       help='Max number of prompts to process at once')
    return parser

def lpr_pather(model, args):
    print_rank_0(model)
    if args.moe_lpr_stage == 1:
        print_rank_0("Fine-tuning method: MoE-LPR stage 1")
        ep_rank = mpu.get_expert_model_parallel_rank()
        if args.enable_shared_expert:
            for name, param in model.named_parameters():
                if "local_experts" in name:
                    param.requires_grad_(False)
                elif "router" in name or 'shared_expert_gate' in name:
                    param.requires_grad_(False)
                else:
                    param.requires_grad_(False)
        else:
            for name, param in model.named_parameters():
                if "local_experts" in name:
                    if ep_rank == 0 and "experts.local_experts.0" in name:
                        param.requires_grad_(False)
                    else:
                        param.requires_grad_(False)
                elif "router" in name:
                    param.requires_grad_(False)
                else:
                    param.requires_grad_(False)
                
    if args.moe_lpr_stage == 2:
        print_rank_0("Fine-tuning method: MoE-LPR stage 2")
        if args.enable_shared_expert:
            for name, param in model.named_parameters():
                if 'shared_expert_gate' in name:
                    param.requires_grad_(False)
                else:
                    param.requires_grad_(False)
        else:
            for name, param in model.named_parameters():
                if 'router' in name:
                    param.requires_grad_(False)
                else:
                    param.requires_grad_(False)

    model.eval()
    trainable_params, all_param = count_parameters(model)
    param_stats = "trainable params: {:,} || all params: {:,} || trainable%: {:.4f}".format(
            trainable_params, all_param, 100 * trainable_params / all_param
        )
    print("[Rank {}] {}".format(torch.distributed.get_rank(), param_stats), flush=True)


def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    print_rank_0('building GPT model ...')
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.use_legacy_models:
        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )
    else: # using core models
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if use_te:
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)

        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base
        )

    if args.moe_lpr_stage in [1, 2]:
        lpr_pather(model, args)

    return model


def main():
    """Main program."""

    # Note: The default args passed here can be overwritten by using appropriate params (check arguments.py file)
    # Micro batch size is not needed to be set by user. (It is calculated based on inference-batch-times-seqlen-threshold argument)
    initialize_megatron(extra_args_provider=add_lm_eval_args,
                        args_defaults={'no_load_rng': True,
                                       'no_load_optim': True,
                                       'micro_batch_size': 1, 
                                       'exit_on_missing_checkpoint': True})

    args = get_args()
    task_manager = lm_eval.tasks.TaskManager()
    task_list = args.task_list.split(',')

    # Set up model and load checkpoint
    model = get_model(model_provider)#, wrap_with_ddp=False)
    load_checkpoint(model, None, None, strict=False)
    
    if len(model) > 1:
        print("not implemented for pipeline parallel models")
        raise NotImplementedError

    model = model[0]
    print(model)

    if args.zero_expert_down_proj:
        print_rank_0("Filling all expert down proj to zero......")
        for name, param in model.named_parameters():
            if 'local_experts' in name and 'linear_fc2' in name:
                param.data.fill_(0)

    tokenizer = get_tokenizer()
    lm_wrapped_model = HarnessWrapper(model, tokenizer)

    results = lm_eval.simple_evaluate( # call simple_evaluate
        model=lm_wrapped_model,
        tasks=task_list,
        num_fewshot=args.num_fewshot,
        task_manager=task_manager,
        batch_size=args.bsz,
        cache_requests=True,
        log_samples=False,
    )

    if mpu.is_pipeline_last_stage() and lm_wrapped_model.rank == 0:
        print(results['results'])
        with open(args.results_path, 'w') as outfile:
            json.dump(results, outfile, indent = 4, default=str)

if __name__ == "__main__":
    main()
