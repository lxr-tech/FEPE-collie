from typing import Optional, Dict, Callable, Sequence, Tuple, Any, List, Iterable

import torch
from torch import nn
import torch.distributed as dist
from deepspeed.runtime.pipe.engine import PipelineEngine
from deepspeed.accelerator import get_accelerator
from transformers.generation.utils import GenerationConfig
from transformers import PreTrainedTokenizerBase
from peft import PeftModel

from collie.module import PipelineGenerationMixin, GPTLMLoss, PipelineModel
from collie.data.dataloader import CollieDataLoader
from collie.utils.rich_progress import f_rich_progress
from collie.log import logger
from collie.config import CollieConfig
from collie.utils import progress, env, setup_ds_engine, BaseProvider, _GenerationStreamer, _MetricsWrapper, BaseMonitor, _MultiMonitors, broadcast_tensor, ColliePadder, auto_param_call
from collie import Evaluator


class EvaluatorForExtrapolation(Evaluator):
    def __init__(self, 
                 loss_fn: Callable = GPTLMLoss(),
                 collate_fn: Optional[Callable] = ColliePadder(),
                 *args,
                 **kwargs):
        self.loss_fn = loss_fn
        super().__init__(collate_fn=collate_fn, *args, **kwargs)
    
    def eval(self, dataloader: Optional[Iterable] = None):
        """
        对数据集进行一次 eval 测试并返回 metric 的结果。需要注意的是如果 ``Evaluator`` 中的 engine 没有初始化，那么默认会自动初始化一个 engine。

        :param dataloader: 用于 eval 的数据集，为 ``Iterable`` 对象 ，当为 ``None`` 时，使用默认的 ``dataset`` 生成的 ``eval_dataloader``
        """
        if self.engine is None:
            self.init_engine()
        if self.server is not None:
            self.server.start()
        if self.eval_dataloader is None:
            self.eval_dataloader = CollieDataLoader(
                self.dataset, self.config.eval_batch_size,
                self.config.gradient_accumulation_steps, shuffle=False,
                collate_fn=self.collate_fn
            )
            self.eval_steps = len(self.eval_dataloader)
        eval_dataloader = self.eval_dataloader
        if dataloader is not None:
            eval_dataloader = dataloader
                    
        torch.distributed.barrier()
        with progress(eval_dataloader, desc="Evaluating Batch: ", disable=env.rank != 0, total=self.eval_steps) as tqbar_batch:
            for batch_idx, batch in enumerate(tqbar_batch):
                tqbar_batch.set_description(f"Evaluating Batch: {batch_idx} / {self.eval_steps}")
                if self.server is not None:
                    self.server.data_provider_handler()
                self.engine.eval()
                if isinstance(self.engine.module, PipelineModel):
                    self.engine.module.forward_type = "eval"
                if isinstance(self.engine.module, PeftModel) and isinstance(self.engine.module.get_base_model(), PipelineModel):
                    self.engine.module.get_base_model().forward_type = "eval"
                with torch.no_grad():
                    batch['past_key_values'] = None
                    result = self.eval_fn(self, batch)
                self.metric_wrapper.update(result)
        with self.monitor as item:
            metric_results = self.metric_wrapper.get_metric()
            for key in list(metric_results.keys()):
                if isinstance(metric_results[key], dict):
                    for k in list(metric_results[key].keys()):
                        metric_results[f"{key}#{k}"] = metric_results[key][k]
                    del metric_results[key]
            item.update({
                "eval_result": metric_results, 
                "global_batch_idx": self.global_batch_idx,
                "mode": "eval"})
        self.metric_wrapper.reset()
        
        if env.local_rank == 0:
            file = open(self.config.file_name, 'a')
            file.write('\t')
            for key, value in metric_results.items():
                file.write("'{}': {}, ".format(key, value))
            file.write('\n')
            file.close()

        if len(metric_results) > 0 and env.rank == 0:  # 如果 metric 不为 None 需要 print 。
            f_rich_progress.print_json(metric_results)
            
        return metric_results
        
    @staticmethod
    @torch.no_grad()
    def eval_fn(evaluator, batch: Tuple) -> Any:
        """一次验证的基本单元

        :param evaluator: 训练器
        :param batch: 一个 batch 的数据，类型为长度为 ``Dict``，格式为：
        
            .. code-block::
            {
                "input_ids": torch.tensor([[1, 100, 100, 2]]),
                "labels": torch.tensor([[1, 100, 100, 2]]),
            }
    
        :return: 一次验证的结果，为 `Dict` 类型，该结果会被传入 `metric` 的 `update` 方法中
        """
        # concat prompt labels for p-tuning
        if evaluator.config.peft_config and evaluator.config.peft_config.peft_type in ["PROMPT_TUNING", "P_TUNING"]:
            batch_size = batch["input_ids"].shape[0]
            if "labels" in batch.keys():
                prefix_labels = torch.full((batch_size, evaluator.config.peft_config.num_virtual_tokens), -100).to(batch["labels"].device)
                batch["labels"] = torch.cat((prefix_labels, batch["labels"]), dim=1)
        if evaluator.config.pp_size > 1:
            if isinstance(evaluator.engine.module, PipelineModel):
                evaluator.engine.module.forward_type = "eval"
            if isinstance(evaluator.engine.module, PeftModel) and isinstance(evaluator.engine.module.get_base_model(), PipelineModel):
                evaluator.engine.module.get_base_model().forward_type = "eval"
            outputs = evaluator.engine.module(**batch)
        else:
            outputs = evaluator.engine(**batch)
        
        ppl = torch.exp(auto_param_call(evaluator.loss_fn, {**batch, **outputs},
                                        signature_fn=evaluator.loss_fn.forward if isinstance(evaluator.loss_fn, nn.Module) else evaluator.loss_fn))
            
        seq_len = torch.sum(batch["attention_mask"], dim=-1) - 1
        
        logits = outputs.get('logits')[:, :-1, :].contiguous()
        target = batch['input_ids'][:, 1:].cuda().contiguous()
        pred = torch.max(logits, dim=-1)[1]

        return {
            'target': target.detach().cuda(), 'pred': pred.detach().cuda(),
            'seq_len': seq_len.cuda(), 'ppl': ppl.detach().clone().view(1,).cuda(),
        }
        
