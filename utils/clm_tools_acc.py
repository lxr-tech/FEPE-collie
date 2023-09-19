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
from collie.utils.seq_len_to_mask import seq_len_to_mask
from collie.log import logger
from collie.config import CollieConfig
from collie.utils import progress, env, setup_ds_engine, BaseProvider, _GenerationStreamer, _MetricsWrapper, BaseMonitor, _MultiMonitors, broadcast_tensor, ColliePadder, auto_param_call
from collie.metrics.base import BaseMetric
from collie import Evaluator

import numpy as np


class FlashGPTLMLoss(torch.nn.Module):

    def __init__(self, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)  # ignore <pad> when compute loss
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        shift_logits = logits[:-1, :].contiguous()
        shift_labels = labels[1:].contiguous().to(logits.device)
        return self.loss(shift_logits, shift_labels)


class CumGPTLMLoss(torch.nn.Module):  # count accumulative perplexity for batch of sequences

    def __init__(self, max_len, ignore_index=-100, cum_enabled=True):
        super().__init__()
        self.max_len = max_len
        self.ignore_index = ignore_index
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().to(logits.device)
        cur_loss = self.loss(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        cur_loss = cur_loss.reshape((-1, self.max_len - 1)).float()
        cum_loss = torch.cumsum(cur_loss, dim=-1)
        cum_loss = cum_loss / torch.arange(1, self.max_len, 1, device='cuda').reshape((1, -1))
        return cum_loss


class EvaluatorForExtrapolation(Evaluator):
    def __init__(self, 
                 loss_fn: Callable = GPTLMLoss(),
                 collate_fn: Optional[Callable] = ColliePadder(),
                 dynamic_enabled=True,
                 dynamic_stride=512,
                 *args,
                 **kwargs):
        self.loss_fn = loss_fn
        self.dynamic_enabled = dynamic_enabled
        self.dynamic_stride = dynamic_stride
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
        
        if env.rank == 0 and self.config.file_name is not None:
            file = open(self.config.file_name, 'a')
            for key, value in metric_results.items():
                file.write('\t"{}": {}, \n'.format(key, value))
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
        if evaluator.dynamic_enabled:
            batch_size, seq_len = batch['input_ids'].shape
            logits = None
            if evaluator.config.pp_size > 1:
                if isinstance(evaluator.engine.module, PipelineModel):
                    evaluator.engine.module.forward_type = "eval"
            for i in range(seq_len, 0, -evaluator.dynamic_stride):
                if env.rank == 0:
                    logger.info(i)
                if evaluator.config.pp_size > 1:
                    outputs = evaluator.engine.module(input_ids=batch['input_ids'][..., :i], 
                                                      labels=batch['labels'][..., :i])
                else:
                    outputs = evaluator.engine(input_ids=batch['input_ids'][..., :i])
                if logits is not None:
                    logits[..., :i, :] = outputs.get('logits')[..., :i, :]
                else:
                    logits = outputs.get('logits')
            outputs = {'logits': logits}
        else:
            if env.rank == 0: 
                logger.info('... evaluating ...')
            if evaluator.config.pp_size > 1:
                if isinstance(evaluator.engine.module, PipelineModel):
                    evaluator.engine.module.forward_type = "eval"
                if isinstance(evaluator.engine.module, PeftModel) and isinstance(evaluator.engine.module.get_base_model(), PipelineModel):
                    evaluator.engine.module.get_base_model().forward_type = "eval"
                outputs = evaluator.engine.module(**batch)
            else:
                outputs = evaluator.engine(**batch)
        
        loss = auto_param_call(evaluator.loss_fn, {**batch, **outputs},
                               signature_fn=evaluator.loss_fn.forward if isinstance(evaluator.loss_fn, nn.Module) else evaluator.loss_fn)
        seq_len = torch.sum((batch['labels'] != 1).int(), dim=-1)
        logits = outputs.get('logits')[..., :-1, :].contiguous()
        target = batch['labels'][..., 1:].cuda().contiguous()
        pred = torch.max(logits, dim=-1)[1]

        return {
            'target': target.detach().float().cpu().numpy(), 'pred': pred.detach().float().cpu().numpy(),
            'loss': loss.detach().float().cpu().numpy(), 'seq_len': seq_len.detach().float().cpu().numpy(), 
        }


class CumPPLMetric(BaseMetric):

    def __init__(self, gather_result: bool = False) -> None:
        super().__init__(gather_result)
        self.loss = []
        self.total = []
        
    def reset(self):
        self.loss = []
        self.total = []
        
    def get_metric(self) -> Optional[Dict]:
        self.loss = np.concatenate(self.loss, axis=0)
        ppl = np.exp(np.mean(self.loss, axis=0))
        return np.round(ppl, 6).tolist()
        
    def update(self, result: Dict):
        assert "loss" in result.keys(), f"loss not in result!"
        loss = result["loss"]
        batch_size, _ = loss.shape
        
        idx = len(self.loss)
        # if env.rank == 0:
        #     file = open(f'./csv_logs/llama2_7B-ntk_dynamic-sample{idx}.json', 'a')
        #     file.write('\t"{}": {}\n'.format('cum#ppl', loss[0].tolist()))
        #     file.write("} \n")
        #     file.close()
        
        self.loss.append(loss)
        self.total.append(batch_size)


class CumAccMetric(BaseMetric):
    """
    计算准确率的 metric

    :param gather_result: 在计算 metric 的时候是否自动将各个进程上的输入进行聚合后再输入到 update 之中。
    """

    def __init__(self, gather_result: bool=False):
        super().__init__(gather_result=gather_result)
        self.correct = []
        self.total = []
    
    def reset(self):
        self.correct = []
        self.total = []

    def get_metric(self)->Dict:
        self.correct = np.concatenate(self.correct, axis=0)
        acc = np.mean(self.correct, axis=0)
        return np.round(acc, 8).tolist()
    
    def update(self, result:Dict):
        r"""
        :meth:`update` 函数将针对一个批次的预测结果做评价指标的累计。

        :param result: 类型为 Dict 且 keys 至少包含["pred", "target"]

            * pred - 预测的 ``torch.Size([B, max_len, n_classes])``
            * target - 真实值 ``torch.Size([B, max_len])``
        """
        assert "pred" in result and "target" in result, "pred and target  must in result, but they not."
        pred = result.get("pred")
        target = result.get("target")
        
        # ddp 时候需要手动 gahter 所有数据。 默认输入的类型都是tensor
        if isinstance(pred, List):
            pred = np.stack(pred, axis=0)
        
        if isinstance(target, List):
            target = np.stack(target, axis=0)
            
        cur_acc = np.cumsum(np.equal(pred, target), axis=-1)
        batch_size, max_len = cur_acc.shape
        cur_acc = cur_acc / np.arange(1, max_len+1, 1).reshape((1, -1))
        
        idx = len(self.correct)
        # if env.rank == 0:
        #     file = open(f'./csv_logs/llama2_7B-ntk_dynamic-sample{idx}.json', 'a')
        #     file.write("{\n")
        #     file.write('\t"{}": {},\n'.format('cum#acc', cur_acc[0].tolist()))
        #     file.close()
        
        self.correct.append(cur_acc)
        self.total.append(batch_size)
