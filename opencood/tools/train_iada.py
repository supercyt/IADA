"""Train fusion-agnostic OPV2V-to-DAIR domain adaptation baselines."""

import argparse
import copy
import glob
import math
import os
import random
import re
import statistics

import numpy as np
import torch
import yaml
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.loss.domain_adaptation_loss import (
    balanced_domain_loss as _generic_balanced_domain_loss,
    compute_adaptation_loss,
    graph_variance_floor_loss as _generic_graph_variance_floor_loss,
)
from opencood.tools import train_utils
from opencood.tools.sim2real_utils import (
    ForeverDataIterator,
    build_prior_encoding,
    build_source_config,
)


DETECTION_OUTPUT_KEYS = ("cls_preds", "reg_preds", "dir_preds")
MODEL_INPUT_KEYS = (
    "processed_lidar",
    "record_len",
    "pairwise_t_matrix",
    "lidar_pose",
)
TRAINING_STATE_FILENAME = "training_state_latest.pth"
SCENE_INDEX_OUTPUT_KEYS = {
    "domain_scene_index",
    "agent_scene_index",
    "ssda_global_scene_index",
    "ssda_local_scene_index",
}


def _atomic_torch_save(value, destination):
    temporary = destination + ".tmp"
    torch.save(value, temporary)
    os.replace(temporary, destination)


def _checkpoint_epoch(path):
    """Return the completed epoch encoded in an OpenCOOD checkpoint name."""

    filename = os.path.basename(path)
    for pattern in (
        r"^net_epoch(\d+)\.pth$",
        r"^net_epoch_bestval_at(\d+)\.pth$",
    ):
        match = re.match(pattern, filename)
        if match:
            return int(match.group(1))
    raise ValueError(f"Unrecognized checkpoint filename: {filename}")


def _find_model_checkpoint(saved_path, prefer_best):
    """Select a model-only checkpoint without silently choosing an old best."""

    if not os.path.isdir(saved_path):
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {saved_path}"
        )

    best_paths = glob.glob(
        os.path.join(saved_path, "net_epoch_bestval_at*.pth")
    )
    if len(best_paths) > 1:
        raise RuntimeError(
            f"Expected at most one best checkpoint in {saved_path}, found "
            f"{len(best_paths)}"
        )

    periodic_paths = []
    for path in glob.glob(os.path.join(saved_path, "net_epoch*.pth")):
        if re.match(r"^net_epoch\d+\.pth$", os.path.basename(path)):
            periodic_paths.append(path)

    if prefer_best and best_paths:
        return best_paths[0]
    if periodic_paths:
        return max(periodic_paths, key=_checkpoint_epoch)
    if best_paths:
        return best_paths[0]
    raise FileNotFoundError(
        f"No net_epoch*.pth checkpoint found in {saved_path}"
    )


def _extract_model_state(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a state-dict mapping")
    if "model" in checkpoint:
        checkpoint = checkpoint["model"]
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint 'model' entry must be a state dict")
    return checkpoint


def _canonical_config_value(value):
    if isinstance(value, np.ndarray):
        return [_canonical_config_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            key: _canonical_config_value(child)
            for key, child in sorted(value.items())
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_config_value(item) for item in value]
    return value


def _adaptation_protocol_signature(hypes):
    model_args = dict(hypes["model"]["args"])
    model_args.pop("interaction_da", None)
    model_args.pop("domain_adapter", None)
    signature = {
        "target_data": {
            key: hypes.get(key)
            for key in (
                "data_dir",
                "root_dir",
                "validate_dir",
                "test_dir",
            )
        },
        "source_domain": hypes["domain_adaptation"]["source"],
        "fusion": hypes["fusion"],
        "preprocess": hypes["preprocess"],
        "postprocess": hypes["postprocess"],
        "model_core_method": hypes["model"]["core_method"],
        "model_args_without_adapter": model_args,
        "max_cav": hypes["train_params"]["max_cav"],
        "comm_range": hypes["comm_range"],
        "noise_setting": hypes.get("noise_setting"),
        "input_source": hypes.get("input_source"),
        "label_type": hypes.get("label_type"),
        # DAIRV2XBaseDataset defaults missing legacy configs to the official
        # DUSA car-only protocol.  An explicit false remains a protocol change.
        "car_only": hypes.get("car_only", True),
    }
    return _canonical_config_value(signature)


def _validate_baseline_warm_start(pretrained_model_dir, current_hypes):
    config_path = os.path.join(pretrained_model_dir, "config.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            "Domain adaptation warm start requires the config.yaml saved by the "
            f"common-geometry baseline: {config_path}"
        )

    with open(config_path, "r") as stream:
        pretrained_hypes = yaml.load(stream, Loader=yaml.Loader)
    parser_name = pretrained_hypes.get("yaml_parser")
    if parser_name is not None:
        parser = getattr(yaml_utils, parser_name, None)
        if parser is None:
            raise ValueError(
                f"Unknown yaml_parser {parser_name!r} in {config_path}"
            )
        pretrained_hypes = parser(pretrained_hypes)
    pretrained_stage = pretrained_hypes.get(
        "domain_adaptation", {}
    ).get("stage")
    if pretrained_stage != "baseline":
        raise ValueError(
            "Every adaptation method must start from the source-only baseline, but "
            f"{config_path} records stage {pretrained_stage!r}."
        )

    if (
        _adaptation_protocol_signature(pretrained_hypes)
        != _adaptation_protocol_signature(current_hypes)
    ):
        raise ValueError(
            "The baseline checkpoint does not use the same OPV2V source, "
            "DAIR target splits, ROI, voxel grid, anchors, fusion setup, "
            "max_cav, noise policy, and detector/fusion architecture as the "
            "current adaptation experiment."
        )


def _load_model_checkpoint(
    model,
    checkpoint_path,
    *,
    allow_missing_adapter,
    require_fresh_adapter=False,
):
    """Load a checkpoint while allowing only a fresh adapter branch."""

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    state_dict = _extract_model_state(checkpoint)
    adapter_prefixes = ("domain_adapter.", "interaction_da.")
    if require_fresh_adapter and any(
        key.startswith(adapter_prefixes) for key in state_dict
    ):
        raise ValueError(
            "The warm-start checkpoint already contains domain-adapter "
            "parameters. Every method must start with a fresh adapter branch "
            "from the same source-only baseline."
        )
    incompatible = model.load_state_dict(state_dict, strict=False)

    unexpected = list(incompatible.unexpected_keys)
    missing = list(incompatible.missing_keys)
    allowed_missing = (
        [key for key in missing if key.startswith(adapter_prefixes)]
        if allow_missing_adapter
        else []
    )
    forbidden_missing = [
        key for key in missing if key not in set(allowed_missing)
    ]
    if unexpected or forbidden_missing:
        raise RuntimeError(
            "Checkpoint is not a compatible detector/fusion model. "
            f"Missing shared keys: {forbidden_missing}; "
            f"unexpected keys: {unexpected}"
        )
    return missing, unexpected


def _capture_random_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_random_state(state):
    if not isinstance(state, dict):
        raise TypeError("Saved random state must be a mapping")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _save_training_state(
    saved_path,
    *,
    stage,
    completed_epoch,
    global_step,
    model,
    optimizer,
    scheduler,
    scaler,
    best_validation_loss,
    best_validation_path,
):
    """Atomically save everything required for an exact epoch-boundary resume."""

    state = {
        "format_version": 1,
        "stage": stage,
        "epoch": int(completed_epoch),
        "global_step": int(global_step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "amp_enabled": bool(scaler.is_enabled()),
        "best_validation_loss": float(best_validation_loss),
        "best_validation_checkpoint": (
            os.path.basename(best_validation_path)
            if best_validation_path
            else None
        ),
        "random_state": _capture_random_state(),
    }
    destination = os.path.join(saved_path, TRAINING_STATE_FILENAME)
    _atomic_torch_save(state, destination)


def _load_training_state(saved_path, model, expected_stage):
    state_path = os.path.join(saved_path, TRAINING_STATE_FILENAME)
    if not os.path.isfile(state_path):
        return None

    state = torch.load(
        state_path, map_location="cpu", weights_only=False
    )
    required = {
        "format_version",
        "stage",
        "epoch",
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "amp_enabled",
        "best_validation_loss",
        "best_validation_checkpoint",
        "random_state",
    }
    if not isinstance(state, dict) or not required.issubset(state):
        available = state if isinstance(state, dict) else {}
        missing = sorted(required.difference(available))
        raise RuntimeError(
            "Incomplete domain-adaptation training state "
            f"{state_path}; missing {missing}"
        )
    if state["format_version"] != 1:
        raise RuntimeError(
            f"Unsupported domain-adaptation training-state version "
            f"{state['format_version']!r}"
        )
    if state["stage"] != expected_stage:
        raise ValueError(
            f"Cannot resume stage {state['stage']!r} as "
            f"{expected_stage!r}. Use --pretrained_model_dir to start a new "
            "stage."
        )
    model.load_state_dict(_extract_model_state(state), strict=True)
    return state


def _archive_orphan_checkpoint(path):
    destination = path + ".orphaned"
    suffix = 1
    while os.path.exists(destination):
        destination = f"{path}.orphaned.{suffix}"
        suffix += 1
    os.rename(path, destination)
    print(
        f"Archived checkpoint left by an interrupted best-model update: "
        f"{destination}"
    )
    return destination


def _resolve_best_validation_path(saved_path, resume_state):
    best_paths = glob.glob(
        os.path.join(saved_path, "net_epoch_bestval_at*.pth")
    )
    if resume_state is None:
        if len(best_paths) > 1:
            raise RuntimeError(
                f"Expected at most one legacy best checkpoint, found "
                f"{len(best_paths)} in {saved_path}"
            )
        return best_paths[0] if best_paths else None

    expected_name = resume_state["best_validation_checkpoint"]
    expected_path = (
        os.path.join(saved_path, expected_name)
        if expected_name is not None
        else None
    )
    if expected_path is not None and not os.path.isfile(expected_path):
        raise RuntimeError(
            "The training state references a missing best checkpoint: "
            f"{expected_path}"
        )

    for path in best_paths:
        if expected_path is None or os.path.abspath(path) != os.path.abspath(
            expected_path
        ):
            _archive_orphan_checkpoint(path)
    return expected_path


def train_parser():
    parser = argparse.ArgumentParser(
        description="OPV2V-to-DAIR collaborative domain adaptation"
    )
    parser.add_argument(
        "--hypes_yaml", "-y", type=str, required=True
    )
    parser.add_argument(
        "--model_dir",
        default="",
        help="Resume an existing domain-adaptation experiment directory.",
    )
    parser.add_argument(
        "--pretrained_model_dir",
        default="",
        help=(
            "Warm start a new method from its matching baseline directory. "
            "Omit it to train the detector and adapter from scratch."
        ),
    )
    parser.add_argument(
        "--stage",
        choices=("baseline", "grl", "dusa", "cudax", "iada", "ssda"),
        default=None,
        help=(
            "baseline: source detection only; grl: naive discriminator; "
            "dusa: LSA+CIA; cudax: CKT+BLC+CPA; iada: graph alignment; "
            "ssda: Selective Shift FSA+SAA"
        ),
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Use CUDA automatic mixed precision.",
    )
    return parser.parse_args()


def _configure_stage(hypes, requested_stage):
    da_cfg = hypes["domain_adaptation"]
    stage = requested_stage or da_cfg.get("stage", "iada")
    adapter_cfg = hypes["model"]["args"].setdefault(
        "domain_adapter", {}
    )

    if stage == "baseline":
        adapter_cfg["enabled"] = False
        adapter_cfg["method"] = "none"
        da_cfg["mode"] = "source_only"
    elif stage in ("grl", "dusa", "cudax", "iada", "ssda"):
        adapter_cfg["enabled"] = True
        adapter_cfg["method"] = stage
        da_cfg["mode"] = "uda"
    else:
        raise ValueError(f"Unsupported training stage: {stage}")

    if stage == "dusa" and "feature_size" not in adapter_cfg:
        raise ValueError(
            "DUSA requires model.args.domain_adapter.feature_size=[H, W]"
        )
    if stage == "cudax":
        configured_counts = {
            int(value)
            for value in (
                adapter_cfg.get("cudax_bin_count"),
                adapter_cfg.get("bin_count"),
                da_cfg.get("cudax_bin_count"),
            )
            if value is not None
        }
        if len(configured_counts) > 1:
            raise ValueError(
                "CUDA-X bin count differs between model and loss config"
            )
        bin_count = configured_counts.pop() if configured_counts else 5
        if bin_count <= 1:
            raise ValueError("CUDA-X bin count must be greater than one")
        adapter_cfg["cudax_bin_count"] = bin_count
        da_cfg["cudax_bin_count"] = bin_count
        residual_bounds = da_cfg.get("cudax_residual_bounds")
        if residual_bounds is None or len(residual_bounds) != 6 or any(
            float(value) <= 0 for value in residual_bounds
        ):
            raise ValueError(
                "CUDA-X requires six positive source-only "
                "cudax_residual_bounds in encoded "
                "[x, y, z, h, w, l] order"
            )

    nonnegative_weights = (
        "domain_loss_weight",
        "graph_variance_weight",
        "graph_variance_target_std",
        "dusa_lsa_weight",
        "dusa_cia_weight",
        "cudax_bin_loss_weight",
        "cudax_domain_loss_weight",
        "ssda_global_weight",
        "ssda_local_weight",
    )
    for key in nonnegative_weights:
        if float(da_cfg.get(key, 0.0)) < 0:
            raise ValueError(f"{key} must be non-negative")

    da_cfg["stage"] = stage
    suffix = f"_{stage}"
    if not hypes["name"].endswith(suffix):
        hypes["name"] += suffix
    return stage


def _make_loader(
    dataset,
    batch_size,
    num_workers,
    shuffle,
    drop_last,
    pin_memory,
):
    loader_args = {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": dataset.collate_batch_train,
        "shuffle": shuffle,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
        "worker_init_fn": _seed_worker,
    }
    if num_workers > 0:
        loader_args["prefetch_factor"] = 2
    return DataLoader(**loader_args)


def _seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _build_clean_validation_config(hypes):
    """Copy a dataset config and disable synthetic localization noise."""

    validation_hypes = copy.deepcopy(hypes)
    noise_setting = validation_hypes.setdefault("noise_setting", {})
    noise_setting["add_noise"] = False
    noise_args = noise_setting.setdefault("args", {})
    for key in ("pos_std", "rot_std", "pos_mean", "rot_mean"):
        noise_args[key] = 0
    return validation_hypes


def _set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_model_inputs(ego_batch, domain="source"):
    missing = [key for key in MODEL_INPUT_KEYS if key not in ego_batch]
    if missing:
        raise KeyError(f"Batch is missing model inputs: {missing}")
    model_inputs = {key: ego_batch[key] for key in MODEL_INPUT_KEYS}
    # Source-only baseline/validation must not infer infrastructure from local
    # index 1: OPV2V collaborators are all vehicles. An existing prior is
    # validated by build_prior_encoding instead of being silently trusted.
    model_inputs["prior_encoding"] = build_prior_encoding(ego_batch, domain)
    return model_inputs


def _merge_domain_outputs(
    source_output, target_output, source_scene_count
):
    """Join adapter tensors from independent source/target forwards.

    Keeping the detector forwards independent prevents source and target
    samples from sharing BatchNorm statistics.  Scene indices produced by the
    target forward are local to that mini-batch, so they are shifted before
    the adaptation loss sees the combined domain batch.
    """

    source_keys = set(source_output) - set(DETECTION_OUTPUT_KEYS)
    target_keys = set(target_output) - set(DETECTION_OUTPUT_KEYS)
    if source_keys != target_keys:
        raise KeyError(
            "Source and target adapter outputs differ: "
            f"source-only={sorted(source_keys - target_keys)}, "
            f"target-only={sorted(target_keys - source_keys)}"
        )
    if not source_keys:
        raise KeyError("Domain adaptation model produced no adapter outputs")

    merged = {}
    for key in source_keys:
        source_value = source_output[key]
        target_value = target_output[key]
        if not torch.is_tensor(source_value) or not torch.is_tensor(
            target_value
        ):
            raise TypeError(f"Adapter output {key!r} must be a tensor")
        if key in SCENE_INDEX_OUTPUT_KEYS:
            target_value = target_value + int(source_scene_count)
        try:
            merged[key] = torch.cat((source_value, target_value), dim=0)
        except RuntimeError as error:
            raise ValueError(
                f"Cannot concatenate source/target adapter output {key!r}"
            ) from error
    return merged


def _slice_detection_output(output_dict, source_scene_count):
    detection_output = {}
    for key in DETECTION_OUTPUT_KEYS:
        if key in output_dict:
            detection_output[key] = output_dict[key][:source_scene_count]
    if "cls_preds" not in detection_output or "reg_preds" not in detection_output:
        raise KeyError("Model output is missing source detection predictions")
    return detection_output


def _balanced_domain_loss(
    domain_logits,
    domain_labels,
    valid_graph_mask,
    source_scene_count,
):
    if source_scene_count < 0 or source_scene_count > domain_labels.numel():
        raise ValueError("source_scene_count is outside the domain batch")
    return _generic_balanced_domain_loss(
        domain_logits,
        domain_labels,
        scene_indices=torch.arange(
            domain_labels.numel(), device=domain_logits.device
        ),
        valid_mask=valid_graph_mask,
    )


def _graph_variance_floor_loss(
    graph_embedding,
    valid_graph_mask,
    source_scene_count,
    target_std,
    epsilon=1.0e-8,
):
    """Keep the interaction representation from collapsing to a constant.

    The RMS standard-deviation floor is computed independently for source and
    target graphs so it cannot be satisfied merely by separating the two
    domains. Using one aggregate statistic preserves information without
    forcing every ReLU channel to become active. A balanced update is applied
    only when both domains contribute at least two valid interaction graphs;
    otherwise a differentiable zero is returned. Setting ``target_std`` to
    zero disables the penalty.
    """

    return _generic_graph_variance_floor_loss(
        graph_embedding,
        valid_graph_mask,
        source_scene_count,
        target_std,
        epsilon,
    )


def _grl_lambda(global_step, total_steps, max_value, gamma):
    if total_steps <= 1:
        progress = 1.0
    else:
        progress = global_step / float(total_steps - 1)
    return max_value * (2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0)


def _setup_stage_optimizer(hypes, model, stage):
    """Configure a uniform scratch LR or discriminative warm-start LRs.

    A scratch run initializes every parameter together and therefore trains
    the detector and adapter at the optimizer base LR. For a baseline-warm-
    started adaptation run, the fresh adapter keeps the base LR while the
    detector uses ``pretrained_lr_scale``. The initialization mode is persisted
    in config.yaml so resume reconstructs the same optimizer parameter groups.
    """

    initialization = hypes["domain_adaptation"].get(
        "initialization",
        "scratch" if stage == "baseline" else "baseline_warm_start",
    )
    if initialization not in ("scratch", "baseline_warm_start"):
        raise ValueError(
            "domain_adaptation.initialization must be 'scratch' or "
            "'baseline_warm_start'"
        )
    if stage == "baseline" or initialization == "scratch":
        return train_utils.setup_optimizer(hypes, model)

    optimizer_cfg = hypes["optimizer"]
    da_cfg = hypes["domain_adaptation"]
    base_lr = float(optimizer_cfg["lr"])
    pretrained_scale = float(da_cfg.get("pretrained_lr_scale", 1.0))
    adapter_scale = float(
        da_cfg.get(
            "adapter_lr_scale",
            da_cfg.get("interaction_lr_scale", 1.0),
        )
    )
    if pretrained_scale < 0 or adapter_scale < 0:
        raise ValueError("optimizer LR scales must be non-negative")

    pretrained_parameters = []
    adapter_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(("domain_adapter.", "interaction_da.")):
            adapter_parameters.append(parameter)
        else:
            pretrained_parameters.append(parameter)

    if not pretrained_parameters:
        raise ValueError("no pretrained model parameters found")
    if not adapter_parameters:
        raise ValueError("no domain-adapter parameters found")

    optimizer_method = getattr(
        torch.optim, optimizer_cfg["core_method"], None
    )
    if optimizer_method is None:
        raise ValueError(
            f"{optimizer_cfg['core_method']} optimizer is not supported"
        )
    parameter_groups = [
        {
            "params": pretrained_parameters,
            "lr": base_lr * pretrained_scale,
            "group_name": "pretrained",
        },
        {
            "params": adapter_parameters,
            "lr": base_lr * adapter_scale,
            "group_name": "adapter",
        },
    ]
    return optimizer_method(
        parameter_groups,
        lr=base_lr,
        **optimizer_cfg.get("args", {}),
    )


@torch.no_grad()
def _validate_detection(model, criterion, data_loader, device, domain):
    if domain not in ("source", "target"):
        raise ValueError("validation domain must be 'source' or 'target'")

    model.eval()
    validation_losses = []

    for batch_data in data_loader:
        if batch_data is None:
            continue
        ego_batch = batch_data["ego"]
        model_inputs = train_utils.to_device(
            _select_model_inputs(ego_batch, domain), device
        )
        model_inputs["grl_lambda"] = 0.0
        label_dict = train_utils.to_device(ego_batch["label_dict"], device)

        output_dict = model(model_inputs)
        detection_output = _slice_detection_output(
            output_dict, int(model_inputs["record_len"].numel())
        )
        validation_loss = criterion(detection_output, label_dict)
        validation_losses.append(float(validation_loss.item()))

    if not validation_losses:
        raise RuntimeError(
            f"{domain.capitalize()} validation loader produced no valid batch"
        )
    return statistics.mean(validation_losses)


def _validate_source(model, criterion, data_loader, device):
    return _validate_detection(
        model, criterion, data_loader, device, domain="source"
    )


def _validate_target(model, criterion, data_loader, device):
    return _validate_detection(
        model, criterion, data_loader, device, domain="target"
    )


def main():
    # Dataset imports pull optional point-cloud readers; keep them on the
    # executable path so helper functions remain importable in lightweight
    # environments and unit tests.
    from opencood.data_utils.datasets import build_dataset

    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    saved_stage = hypes["domain_adaptation"].get("stage")
    if (
        opt.model_dir
        and opt.stage is not None
        and saved_stage is not None
        and opt.stage != saved_stage
    ):
        raise ValueError(
            f"--model_dir resumes the saved stage {saved_stage!r}; it cannot "
            f"be changed to {opt.stage!r}. Use --pretrained_model_dir for a "
            "new adaptation stage."
        )
    stage = _configure_stage(hypes, opt.stage)
    da_cfg = hypes["domain_adaptation"]
    pretrained_model_dir = ""
    if opt.model_dir:
        # Runs created before this field existed could only be warm-started for
        # adaptation stages, so that is the safe legacy resume default.
        initialization = da_cfg.get(
            "initialization",
            "scratch" if stage == "baseline" else "baseline_warm_start",
        )
    else:
        pretrained_model_dir = (
            opt.pretrained_model_dir
            or da_cfg.get("pretrained_model_dir", "")
        )
        initialization = (
            "baseline_warm_start" if pretrained_model_dir else "scratch"
        )
        if stage != "baseline" and pretrained_model_dir:
            _validate_baseline_warm_start(pretrained_model_dir, hypes)
    da_cfg["initialization"] = initialization

    _set_random_seed(int(da_cfg.get("seed", 303)))
    source_hypes = build_source_config(hypes)
    source_validate_hypes = _build_clean_validation_config(source_hypes)
    target_validate_hypes = _build_clean_validation_config(hypes)

    print(f"Training stage: {stage}")
    print(f"Initialization: {initialization}")
    print("Building source OPV2V datasets")
    source_train_dataset = build_dataset(
        source_hypes, visualize=False, train=True
    )
    source_validate_dataset = build_dataset(
        source_validate_hypes, visualize=False, train=False
    )

    use_target = stage != "baseline"
    target_train_dataset = None
    target_validate_dataset = None
    if use_target:
        print(
            "Building target DAIR-V2X dataset "
            "(target labels are excluded from model inputs and training losses)"
        )
        target_train_dataset = build_dataset(
            hypes, visualize=False, train=True
        )
        target_validate_dataset = build_dataset(
            target_validate_hypes, visualize=False, train=False
        )

    source_batch_size = int(da_cfg.get("source_batch_size", 2))
    target_batch_size = int(da_cfg.get("target_batch_size", 2))
    num_workers = int(da_cfg.get("num_workers", 4))
    pin_memory = torch.cuda.is_available()

    source_train_loader = _make_loader(
        source_train_dataset,
        source_batch_size,
        num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=pin_memory,
    )
    source_validate_loader = _make_loader(
        source_validate_dataset,
        source_batch_size,
        num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=pin_memory,
    )
    target_train_loader = None
    target_validate_loader = None
    if use_target:
        target_train_loader = _make_loader(
            target_train_dataset,
            target_batch_size,
            num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=pin_memory,
        )
        target_validate_loader = _make_loader(
            target_validate_dataset,
            target_batch_size,
            num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=pin_memory,
        )

    print("Creating collaborative domain-adaptation model")
    model = train_utils.create_model(hypes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resume_state = None
    legacy_resume = False
    legacy_resume_checkpoint_path = None

    if opt.model_dir:
        saved_path = opt.model_dir
        resume_state = _load_training_state(saved_path, model, stage)
        if resume_state is not None:
            init_epoch = int(resume_state["epoch"])
            print(
                f"Resuming complete training state from epoch {init_epoch}"
            )
        else:
            legacy_resume = True
            checkpoint_path = _find_model_checkpoint(
                saved_path, prefer_best=False
            )
            legacy_resume_checkpoint_path = checkpoint_path
            _load_model_checkpoint(
                model,
                checkpoint_path,
                allow_missing_adapter=False,
            )
            init_epoch = _checkpoint_epoch(checkpoint_path)
            print(
                "Warning: legacy model-only checkpoint found; optimizer, "
                "scheduler, scaler, and random state cannot be restored."
            )
            print(
                f"Resuming model weights from epoch {init_epoch}: "
                f"{checkpoint_path}"
            )
    else:
        init_epoch = 0
        if pretrained_model_dir:
            checkpoint_path = _find_model_checkpoint(
                pretrained_model_dir, prefer_best=True
            )
            missing, _ = _load_model_checkpoint(
                model,
                checkpoint_path,
                allow_missing_adapter=True,
                require_fresh_adapter=stage != "baseline",
            )
            print(
                f"Warm started from {checkpoint_path}; initialized "
                f"{len(missing)} adapter-only parameters from scratch"
            )
        saved_path = train_utils.setup_train(hypes)

    model.to(device)
    criterion = train_utils.create_loss(hypes)
    optimizer = _setup_stage_optimizer(hypes, model, stage)
    scheduler = train_utils.setup_lr_schedular(
        hypes,
        optimizer,
        init_epoch=0 if resume_state is not None else init_epoch,
    )
    writer = SummaryWriter(saved_path)

    amp_enabled = bool(opt.half and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    if resume_state is not None:
        if bool(resume_state["amp_enabled"]) != amp_enabled:
            raise ValueError(
                "AMP mode differs from the saved training state. Resume with "
                "the same --half setting and a compatible CUDA device."
            )
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        scaler.load_state_dict(resume_state["scaler"])
        _restore_random_state(resume_state["random_state"])
        # The checkpoint captures RNG state immediately before the next
        # epoch's reinitialization, so reconstruct the same dataset ordering.
        source_train_dataset.reinitialize()
        if target_train_dataset is not None:
            target_train_dataset.reinitialize()

    epochs = int(hypes["train_params"]["epoches"])
    configured_steps = da_cfg.get("steps_per_epoch")
    steps_per_epoch = (
        int(configured_steps)
        if configured_steps is not None
        else len(source_train_loader)
    )
    if steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be positive")

    total_steps = max(epochs * steps_per_epoch, 1)
    grl_cfg = da_cfg.get("grl", {})
    grl_max = float(grl_cfg.get("max_value", 1.0))
    grl_gamma = float(grl_cfg.get("gamma", 10.0))
    gradient_clip = float(da_cfg.get("gradient_clip_norm", 0.0))
    log_freq = int(da_cfg.get("log_freq", 10))

    best_validation_loss = (
        float(resume_state["best_validation_loss"])
        if resume_state is not None
        else float("inf")
    )
    best_validation_path = _resolve_best_validation_path(
        saved_path, resume_state
    )
    if legacy_resume and best_validation_path is not None:
        _load_model_checkpoint(
            model,
            best_validation_path,
            allow_missing_adapter=False,
        )
        best_validation_loss = _validate_source(
            model, criterion, source_validate_loader, device
        )
        _load_model_checkpoint(
            model,
            legacy_resume_checkpoint_path,
            allow_missing_adapter=False,
        )
        print(
            "Recovered legacy best source-validation loss: "
            f"{best_validation_loss:.4f}"
        )

    print(f"Training outputs: {saved_path}")
    for epoch in range(init_epoch, max(epochs, init_epoch)):
        superseded_best_path = None
        model.train()
        source_iterator = ForeverDataIterator(source_train_loader)
        target_iterator = (
            ForeverDataIterator(target_train_loader) if use_target else None
        )

        for step in range(steps_per_epoch):
            source_batch = next(source_iterator)
            if source_batch is None:
                continue
            source_ego = source_batch["ego"]

            if use_target:
                target_batch = next(target_iterator)
                if target_batch is None:
                    continue
                target_ego = target_batch["ego"]
                source_model_inputs = _select_model_inputs(
                    source_ego, "source"
                )
                target_model_inputs = _select_model_inputs(
                    target_ego, "target"
                )
                source_scene_count = int(
                    source_ego["record_len"].numel()
                )
                target_scene_count = int(
                    target_ego["record_len"].numel()
                )
                domain_labels = torch.cat(
                    (
                        torch.zeros(source_scene_count),
                        torch.ones(target_scene_count),
                    )
                )
            else:
                source_model_inputs = _select_model_inputs(source_ego)
                source_scene_count = int(
                    source_ego["record_len"].numel()
                )
                domain_labels = None

            global_step = epoch * steps_per_epoch + step
            current_grl = (
                _grl_lambda(
                    global_step,
                    total_steps,
                    grl_max,
                    grl_gamma,
                )
                if use_target
                else 0.0
            )

            source_model_inputs = train_utils.to_device(
                source_model_inputs, device
            )
            source_model_inputs["grl_lambda"] = current_grl
            if use_target:
                target_model_inputs = train_utils.to_device(
                    target_model_inputs, device
                )
                target_model_inputs["grl_lambda"] = current_grl
            source_label_dict = train_utils.to_device(
                source_ego["label_dict"], device
            )
            if domain_labels is not None:
                domain_labels = domain_labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                source_output = model(source_model_inputs)
                detection_output = _slice_detection_output(
                    source_output, source_scene_count
                )
                detection_loss = criterion(
                    detection_output, source_label_dict
                )

                if use_target:
                    target_output = model(target_model_inputs)
                    output_dict = _merge_domain_outputs(
                        source_output,
                        target_output,
                        source_scene_count,
                    )
                    record_len = torch.cat(
                        (
                            source_model_inputs["record_len"],
                            target_model_inputs["record_len"],
                        )
                    )
                    adaptation_loss, adaptation_metrics = (
                        compute_adaptation_loss(
                            stage,
                            output_dict,
                            domain_labels,
                            source_scene_count,
                            record_len,
                            source_label_dict,
                            da_cfg,
                        )
                    )
                    domain_loss = adaptation_metrics.get(
                        "domain_loss", adaptation_loss
                    )
                    domain_accuracy = adaptation_metrics.get(
                        "domain_accuracy",
                        detection_loss.new_tensor(float("nan")),
                    )
                    total_loss = detection_loss + adaptation_loss
                    if "graph_embedding" in output_dict:
                        graph_embedding = output_dict["graph_embedding"]
                        adaptation_metrics["graph_embedding_norm"] = (
                            graph_embedding.norm(dim=1).mean()
                        )
                        adaptation_metrics["graph_embedding_variance"] = (
                            graph_embedding.var(dim=0, unbiased=False).mean()
                        )
                else:
                    adaptation_loss = detection_loss.new_zeros(())
                    adaptation_metrics = {}
                    domain_loss = adaptation_loss
                    domain_accuracy = detection_loss.new_tensor(float("nan"))
                    total_loss = detection_loss

            scaler.scale(total_loss).backward()
            if gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), gradient_clip
                )
            scaler.step(optimizer)
            scaler.update()

            writer.add_scalar(
                "Train/total_loss", total_loss.item(), global_step
            )
            writer.add_scalar(
                "Train/detection_loss",
                detection_loss.item(),
                global_step,
            )
            writer.add_scalar(
                "Train/adaptation_loss",
                adaptation_loss.item(),
                global_step,
            )
            writer.add_scalar("Train/grl_lambda", current_grl, global_step)
            for metric_name, metric_value in adaptation_metrics.items():
                if not isinstance(metric_value, torch.Tensor) or (
                    metric_value.numel() != 1
                ):
                    continue
                if torch.isfinite(metric_value).item():
                    writer.add_scalar(
                        "Train/{}".format(metric_name),
                        metric_value.item(),
                        global_step,
                    )

            if step % log_freq == 0:
                accuracy_text = (
                    f"{domain_accuracy.item():.3f}"
                    if not torch.isnan(domain_accuracy)
                    else "n/a"
                )
                print(
                    f"[epoch {epoch}][{step + 1}/{steps_per_epoch}] "
                    f"total={total_loss.item():.4f} "
                    f"det={detection_loss.item():.4f} "
                    f"domain={domain_loss.item():.4f} "
                    f"domain_acc={accuracy_text} "
                    f"grl={current_grl:.4f}"
                )

        if epoch % int(hypes["train_params"]["eval_freq"]) == 0:
            source_validation_loss = _validate_source(
                model, criterion, source_validate_loader, device
            )
            writer.add_scalar(
                "Validate/source_loss",
                source_validation_loss,
                epoch,
            )
            print(
                f"Epoch {epoch}: source validation loss "
                f"{source_validation_loss:.4f}"
            )

            if target_validate_loader is not None:
                target_validation_loss = _validate_target(
                    model, criterion, target_validate_loader, device
                )
                writer.add_scalar(
                    "Validate/target_loss",
                    target_validation_loss,
                    epoch,
                )
                print(
                    f"Epoch {epoch}: target validation loss "
                    f"{target_validation_loss:.4f} "
                    "(logging only; does not select the best checkpoint)"
                )

            if source_validation_loss < best_validation_loss:
                best_validation_loss = source_validation_loss
                new_best_path = os.path.join(
                    saved_path,
                    f"net_epoch_bestval_at{epoch + 1}.pth",
                )
                _atomic_torch_save(model.state_dict(), new_best_path)
                if (
                    best_validation_path
                    and best_validation_path != new_best_path
                    and os.path.exists(best_validation_path)
                ):
                    superseded_best_path = best_validation_path
                best_validation_path = new_best_path

        if epoch % int(hypes["train_params"]["save_freq"]) == 0:
            _atomic_torch_save(
                model.state_dict(),
                os.path.join(saved_path, f"net_epoch{epoch + 1}.pth"),
            )

        scheduler.step()
        _save_training_state(
            saved_path,
            stage=stage,
            completed_epoch=epoch + 1,
            global_step=(epoch + 1) * steps_per_epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_validation_loss=best_validation_loss,
            best_validation_path=best_validation_path,
        )
        if (
            superseded_best_path is not None
            and os.path.exists(superseded_best_path)
        ):
            os.remove(superseded_best_path)
        source_train_dataset.reinitialize()
        if target_train_dataset is not None:
            target_train_dataset.reinitialize()

    writer.close()
    print(f"Training finished. Checkpoints saved to {saved_path}")
    print(
        "Evaluate on DAIR-V2X with: "
        f"python opencood/tools/inference.py --model_dir {saved_path} "
        "--fusion_method intermediate"
    )


if __name__ == "__main__":
    main()
