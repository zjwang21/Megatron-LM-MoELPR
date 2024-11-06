import os
import re
import torch
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM

class ALPACA:
    def __init__(self, model, tokenizer, save_path, bsz, lang='en', max_len=2048) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.lang = lang
        self.max_len = max_len
        self.bsz = bsz
        self.res_info = []
        self.save_path = save_path
        self.should_log = True
        self.data_path = "./data/{}_alpaca_eval.json".format(lang)

    def prepare_batch(self, batch_q):
        #prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction:\n{}\n\n### Response:\n"
        #texts = [prompt.format(k) for k in batch_q]
        return self.tokenizer(batch_q, return_tensors='pt', padding=True)

    def log(self, batch):
        print("inputs ids:\n{}".format(batch['input_ids']))
        print("inputs:\n{}".format(self.tokenizer.batch_decode(batch['input_ids'])))

    @torch.inference_mode()
    def gen_batch(self, batch):
        output_ids = self.model.generate(
            **batch, 
            generation_config=GenerationConfig(
                max_new_tokens=600,
                do_sample=False,
                temperature=0.0,  # t=0.0 raise error if do_sample=True
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                early_stopping=True,
            )
        ).tolist()
        length = batch['input_ids'].size(-1)
        #print(output_ids)
        output_strs = self.tokenizer.batch_decode([k[length:] for k in output_ids], skip_special_tokens=True)
        return output_strs

    def eval(self):
        data = json.load(open(self.data_path))
        for i in tqdm(range(0, len(data), self.bsz)):
            samples = data[i:i+self.bsz]
            batch_q = [k['{}_instruction'.format(self.lang)] for k in samples]
            batch_a = [k['output'] for k in samples]
            batch = self.prepare_batch(batch_q)
            batch = {k:v.to(self.model.device) for k, v in batch.items()}
            if self.should_log:
                self.log(batch)
                self.should_log = False
            batch_res = self.gen_batch(batch)
            for q, response in zip(batch_q, batch_res):
                self.res_info.append({"instruct": q, "response": response})
                print(self.res_info[-1])
        with open(self.save_path, 'w') as f:
            json.dump(self.res_info, f, indent=2, ensure_ascii=False)

path = "/root/work/huangxin/nanda/models/Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(path)
tokenizer.padding_side='left'
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)
model.cuda().eval()

evaluator = ALPACA(model, tokenizer, "./results/qwen2.5_instruct_en_alpaca_eval.json", 128)
evaluator.eval()