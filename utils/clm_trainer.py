import time
import tqdm
from itertools import chain
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import math
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from transformers import Trainer
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.pytorch_utils import is_torch_greater_or_equal_than_1_10
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import nested_numpify, nested_concat
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    FSDPOption,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
from transformers.utils import (
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    can_return_loss,
    find_labels,
    get_full_repo_name,
    is_accelerate_available,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_ipex_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_compile_available,
    is_torch_neuroncore_available,
    is_torch_tpu_available,
    logging,
    strtobool,
)

logger = logging.get_logger(__name__)

_is_native_cpu_amp_available = is_torch_greater_or_equal_than_1_10

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met


class TrainerForCausalLM(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        # target = inputs['input_ids']
        outputs = model(**inputs)
        loss = outputs.get('loss').mean()
        # logits = outputs.get('logits')
        # logits = logits[:, :-1].contiguous()
        # target = target[:, 1:].contiguous()
        # batch_size, seq_len, vocab_size = logits.shape
        # loss = f.cross_entropy(logits.reshape(-1, vocab_size), target.reshape(-1), ignore_index=0)
        return (loss, outputs) if return_outputs else loss

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None

        if self.control.should_evaluate:
            rank = torch.distributed.get_rank()
            if rank == 0:
                print('\n')

            if isinstance(self.eval_dataset, dict):
                metrics = {}
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    dataset_metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
                    metrics.update(dataset_metrics)
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            if self.state.epoch is not None:
                metrics["epoch"] = round(self.state.epoch, 2)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    # return a dict, metric_key_prefix=f"eval_{eval_dataset_name}" default "eval"
    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        try:
            eval_seq_len = int(metric_key_prefix.split('_')[-1])
        except:
            eval_seq_len = len(eval_dataset[0]['input_ids'])

        eval_batch_size = min(int(self.args.eval_batch_size), max(1,
                              int(512 * self.args.eval_batch_size / eval_seq_len)))

        eval_sampler = self._get_eval_sampler(eval_dataset)

        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

        all_acc, all_ppl = None, None

        rank = torch.distributed.get_rank()

        # if rank == 0:
        #     print(f"***** Running {metric_key_prefix} *****")

        with tqdm.tqdm(eval_dataloader, disable=True) as tqb:
            self.model.eval()
            for batch in tqb:
                with torch.no_grad():
                    acc, ppl = self.eval_step(batch)
                    all_acc = acc if all_acc is None else all_acc + acc
                    all_ppl = ppl if all_ppl is None else all_ppl + ppl

            all_acc_gather = [None for _ in range(self.args.world_size)]
            all_ppl_gather = [None for _ in range(self.args.world_size)]
            torch.distributed.all_gather_object(all_acc_gather, all_acc)
            torch.distributed.all_gather_object(all_ppl_gather, all_ppl)
            all_acc_merged = list(chain(*all_acc_gather))
            all_ppl_merged = list(chain(*all_ppl_gather))
            all_acc = round(sum(all_acc_merged) / len(all_acc_merged), 6)
            all_ppl = round(sum(all_ppl_merged) / len(all_ppl_merged), 6)

        if rank == 0:
            print(f'\'{metric_key_prefix}_acc\':', all_acc, ',', f'\'{metric_key_prefix}_ppl\':', all_ppl, ',\n')

        # start_time = time.time()
        #
        # eval_loop = self.evaluation_loop
        # output = eval_loop(
        #     eval_dataloader,
        #     description="Evaluation",
        #     # No point gathering the predictions if there are no metrics, otherwise we defer to
        #     # self.args.prediction_loss_only
        #     prediction_loss_only=True if self.compute_metrics is None else None,
        #     ignore_keys=ignore_keys,
        #     metric_key_prefix=metric_key_prefix,
        # )
        #
        # total_batch_size = self.args.eval_batch_size * self.args.world_size
        # if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
        #     start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        # output.metrics.update(
        #     speed_metrics(
        #         metric_key_prefix,
        #         start_time,
        #         num_samples=output.num_samples,
        #         num_steps=math.ceil(output.num_samples / total_batch_size),
        #     )
        # )
        #
        # self.log({f'{metric_key_prefix}_acc': all_acc, f'{metric_key_prefix}_ppl': all_ppl})
        #
        # if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
        #     # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
        #     xm.master_print(met.metrics_report())
        #
        # self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)
        #
        # self._memory_tracker.stop_and_update_metrics(output.metrics)

        return {f'{metric_key_prefix}_acc': all_acc, f'{metric_key_prefix}_ppl': all_ppl}

    def eval_step(self, batch):
        self.model.eval()
        # print(self.model)
        outputs = self.model(input_ids=batch['input_ids'].cuda(),
                             attention_mask=batch['attention_mask'].cuda(),
                             labels=batch['labels'].cuda(), )
        loss = outputs.get('loss')
        logits = outputs.get('logits')[:, :-1, :].contiguous()
        target = batch['input_ids'][:, 1:].cuda().contiguous()
        pred = torch.max(logits, dim=-1)[1]

        acc = nested_numpify(torch.mean(pred.eq(target).float(), dim=-1)).tolist()
        ppl = nested_numpify(torch.exp(loss)).tolist()

        return acc, ppl
