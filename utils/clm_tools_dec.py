import torch
import torch.distributed as dist

from transformers.generation.utils import GenerationConfig
from peft import PeftModel

from collie import Evaluator
from collie.data.dataloader import CollieDataLoader
from collie.metrics.base import BaseMetric
from collie.module import PipelineModel, PipelineGenerationMixin
from collie.utils import env, progress, setup_ds_engine
from collie.log.logger import logger

import json
from typing import Any, Dict, Tuple, Optional, Iterable
from collie.utils.rich_progress import f_rich_progress


class MetricsWrapper:
    """
    对输入的 metrics 进行封装以便于支持 Trainer 使用

    :param metrics: 用户传进来的 metric, 类型为 Dict。例如 {'name1': metric1, 'name2': metric2}
    :param trainer: 类型为 :class:`~collie.controller.Trainer`, 用以对 metrics 进行初始化。
    """

    def __init__(self, metrics, trainer):
        self._metrics = []
        self._metric_names = []
        if metrics is not None:
            if not isinstance(metrics, Dict):
                raise TypeError('Parameter `metrics` can only be `Dict` type.')
            for metric_name, metric in metrics.items():
                if isinstance(metric, BaseMetric):
                    metric.construct(trainer)
                else:
                    raise ValueError(f"{metric_name}:{metric.__class__.__name__} must be instance of BaseMetric, but it not!")
                self._metric_names.append(metric_name)
                self._metrics.append(metric)

    def update(self, result):
        """
        针对一个批次的预测结果做评价指标的累计。

        :param result: 用以计算 metric 的预测结果，其类型只能为 Dict。
        """
        for metric in self._metrics:
            if not isinstance(result, dict):
                raise RuntimeError(
                    'The output of your model is of type:`{}`, please '
                    'either directly return a dict from your model'.
                    format(type(result)))
            # gather 输入
            if metric.gather_result:
                gather_out = metric.gather(result)
            else:
                gather_out = result
            metric.update(gather_out)

    def reset(self):
        """
        将 Metric 中的状态重新设置。
        """
        for metric in self._metrics:
            metric.reset()

    def get_metric(self) -> Dict:
        """调用各个 metric 得到 metric 的结果。

        :return: 返回的类型如 {'metric_name1': metric_results,'metric_name2': metric_results}
        """
        results = {}
        for metric_name, metric in zip(self._metric_names, self._metrics):
            if isinstance(metric, BaseMetric):
                _results = metric.get_metric(metric_name)
            else:
                raise RuntimeError(f'Not support `{type(metric)}` for now.')
            if _results is not None:
                results[metric_name] = _results
            else:
                logger.warning_once(f'Metric:{metric_name} returns None when '
                                    'getting metric results.')
        return results


class EvaluatorForGeneration(Evaluator):
    def __init__(
        self,
        generation_config: GenerationConfig = GenerationConfig(),
        skip_special_tokens: bool = True,
        *args,
        **kwargs,
    ):
        self.generation_config = generation_config
        self.skip_special_tokens = skip_special_tokens
        super().__init__(*args, **kwargs)
        self.metric_wrapper = MetricsWrapper(self.metrics, self)

    def init_engine(self):
        """
        初始化 engine。config 中 的 optimizer 手动删掉， 不然会自动调用
        """
        if (
            dist.get_world_size()
            != self.config.tp_size * self.config.dp_size * self.config.pp_size
        ):
            logger.rank_zero_warning(
                "The world size is not equal to the product of the parallel sizes set."
                f"{dist.get_world_size()} != {self.config.tp_size} * {self.config.dp_size} * {self.config.dp_size}."
            )
            self.config.dp_size = dist.get_world_size() // (
                self.config.tp_size * self.config.pp_size
            )
            logger.rank_zero_warning(f"Set dp_size to {self.config.dp_size}.")
        object.__setattr__(
            self, "engine", setup_ds_engine(config=self.config, model=self.model)[0]
        )
        if isinstance(self.engine.module, PipelineGenerationMixin):
            self.engine.module.set_engine(self.engine)
        if isinstance(self.engine.module, PeftModel) and isinstance(
            self.engine.module.get_base_model(), PipelineGenerationMixin
        ):
            self.engine.module.get_base_model().set_engine(self.engine)

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
                self.dataset,
                self.config.eval_batch_size,
                self.config.gradient_accumulation_steps,
                shuffle=False,
                collate_fn=self.collate_fn,
                num_workers=self.config.dataloader_num_workers,
            )
            self.eval_steps = len(self.eval_dataloader)
        eval_dataloader = self.eval_dataloader
        if dataloader is not None:
            eval_dataloader = dataloader
        with progress(
            eval_dataloader,
            desc="Evaluating Batch: ",
            disable=env.rank != 0,
            total=self.eval_steps,
        ) as tqbar_batch:
            for batch_idx, batch in enumerate(tqbar_batch):
                # logger.info(f"batch_idx {batch_idx}, env.rank {env.rank}, env.dp_rank {env.dp_rank}, env.pp_size {env.pp_rank}, env.tp_size {env.tp_rank}, ")
                tqbar_batch.set_description(
                    f"Evaluating Batch: {batch_idx} / {self.eval_steps}"
                )
                if self.server is not None:
                    self.server.data_provider_handler()
                self.engine.eval()
                if isinstance(self.engine.module, PipelineModel):
                    self.engine.module.forward_type = "eval"
                if isinstance(self.engine.module, PeftModel) and isinstance(
                    self.engine.module.get_base_model(), PipelineModel
                ):
                    self.engine.module.get_base_model().forward_type = "eval"
                with torch.no_grad():
                    batch["past_key_values"] = None
                    result = self.eval_fn(self, batch)
                self.metric_wrapper.update(result)
        with self.monitor as item:
            metric_results = self.metric_wrapper.get_metric()
            for key in list(metric_results.keys()):
                if isinstance(metric_results[key], dict):
                    for k in list(metric_results[key].keys()):
                        metric_results[f"{key}#{k}"] = metric_results[key][k]
                    del metric_results[key]
            item.update(
                {
                    "eval_result": metric_results,
                    "global_batch_idx": self.global_batch_idx,
                    "mode": "eval",
                }
            )
        self.metric_wrapper.reset()

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
                "taregt": torch.tensor([[1, 100, 100, 2]]),
            }

        :return: 一次验证的结果，为 `Dict` 类型，该结果会被传入 `metric` 的 `update` 方法中
        """
        if isinstance(evaluator.engine.module, PipelineModel):
            evaluator.engine.module.forward_type = "generate"
        if isinstance(evaluator.engine.module, PeftModel) and isinstance(
            evaluator.engine.module.get_base_model(), PipelineModel
        ):
            evaluator.engine.module.get_base_model().forward_type = "generate"
        assert (
            evaluator.tokenizer is not None
        ), "You must provide a tokenizer to decode the generated results."
        generated_ids = evaluator.engine.module.generate(
            **{k: v for k, v in batch.items() if k in ("input_ids", "attention_mask")},
            generation_config=evaluator.generation_config,
        )
        prompt_length = batch["input_ids"].shape[1]
        result = {
            "pred": [
                evaluator.tokenizer.decode(
                    sample[prompt_length:],
                    skip_special_tokens=evaluator.skip_special_tokens,
                )
                for sample in generated_ids
            ]
        }
        if "target" in batch.keys():
            result["target"] = [
                evaluator.tokenizer.decode(
                    sample, skip_special_tokens=evaluator.skip_special_tokens
                )
                for sample in batch["target"]
            ]
        return result


class PrintMetric(BaseMetric):
    """
    用以保存并打印 decode 生成内容的 metric

    :param verbose: 控制是否使用 logger 打印生成的 sentences
    :param save_to_file: 控制是否保存生成的 sentences 到文件夹中。
    :param save_path: 保存 decode 生成的 sentences 的文件路径, 当 save_to_file 为 `True` 才生效
    """
    def __init__(self, 
                 verbose: bool = True,
                 save_to_file: bool = False,
                 save_path: str = None,
                 gather_result: bool = True) -> None:
        super().__init__(gather_result)
        self.verbose = verbose
        self.save_to_file = save_to_file
        self.save_path = save_path
        self.json = {}
        self.idx = 0 
    
    def get_metric(self, metric_name):
        if env.dp_rank == 0:
            with open(f"{self.save_path}-{metric_name}.json", 'w') as fp:
                json.dump(self.json, fp, indent=4)

    def update(self, result: Dict):
        assert "pred" in result, "result must contain key `pred`"
        assert "target" in result, "result must contain key `target`"
        if (env.dp_rank == 0 and self.gather_result) and (env.pp_rank == env.pp_size-1 and env.tp_rank == env.tp_size-1):
            if self.verbose:
                logger.info(result["pred"])
            if self.save_to_file:
                for i in range(len(result["pred"])):
                    logger.info(self.idx)
                    # logger.info(f"self.idx {self.idx}, env.rank {env.rank}, env.dp_rank {env.dp_rank}, env.pp_rank {env.pp_rank}, env.tp_rank {env.tp_rank}")
                    self.json[str(self.idx)] = {'pred': result['pred'][i], 
                                                'target': result['target'][i], }
                    self.idx += 1
