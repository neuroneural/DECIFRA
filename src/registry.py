"""
Dynamic resolution of models and datasets, and assembly of the effective model
config, from a composed Hydra config.

All helpers take the full composed ``cfg`` and derive what they need from it:
  - ``cfg.mode``         -> "pretrain" | "finetune" (set by the root config)
  - ``cfg.model``        -> the selected model group (architecture + task blocks)
  - ``cfg.dataset``      -> the selected dataset group
  - ``cfg.data_info``    -> runtime sizes, filled in by the script after loading data

Models
------
A model class is imported from ``src.models.<module>``. The class is chosen by:
  - ``variant`` (when set and != "default"): class = ``f"{module}_{variant}"``.
    This lets architectural variants that share one config (and one module file)
    live in a single config file, e.g. ``module: DECIFRA`` + ``variant: IMix`` ->
    class ``DECIFRA_IMix`` in ``src.models.DECIFRA``.
  - otherwise the explicit ``model_name`` is used (defaulting to ``module``).
HPs are canonical in the yaml; ``default_HPs``/``custom_HPs`` are no longer used.

A model config carries shared architecture keys plus optional task blocks::

    module: DECIFRA
    variant: default           # default | noGate | IMix | IMix_Res | ...
    rnn: {...}                 # shared architecture
    pretrain:  {pretraining: true,  loss: {...}}
    finetune:  {loss: {...}, pretrained: {run, checkpoint, load, drop_keys}}

``build_model_cfg`` flattens the block named by ``cfg.mode`` onto the shared keys
and stamps the resolved ``model_name``/``module`` (and, when fine-tuning, the
``pretrained`` source) so the saved config self-describes.

Datasets
--------
A dataset ``bar`` is resolved from ``src.datasets.bar`` and must expose
``load_pretraining_data() -> data`` or ``-> (data, extra_train_data)``.
"""

import os
from importlib import import_module

from omegaconf import OmegaConf


# Keys that are task blocks rather than shared model params.
_TASK_BLOCKS = ("pretrain", "finetune")


def _class_and_module(model_group):
    """
    Resolve ``(class_name, module)`` from a model group config.

    A ``variant`` (set and != "default") selects the class as
    ``f"{module}_{variant}"`` so several classes can share one config file.
    Otherwise the explicit ``model_name`` is used, defaulting to ``module``.
    """
    module = model_group.get("module") or model_group.get("model_name")
    if module is None:
        raise ValueError("model config must define 'module' (or 'model_name').")
    variant = model_group.get("variant", None)
    if variant is not None and str(variant) != "default":
        class_name = f"{module}_{variant}"
    else:
        class_name = model_group.get("model_name") or module
    return class_name, module


def build_model_cfg(cfg):
    """
    Assemble the effective, flat model config the model class consumes.

    Base = shared params + the ``pretrain`` block; when ``cfg.mode == 'finetune'``
    the optional ``finetune`` block is merged on top as a delta. ``pretraining``
    is derived from the mode. Runtime sizes come from ``cfg.data_info``.
    """
    mode = cfg.mode
    shared = {k: v for k, v in cfg.model.items() if k not in _TASK_BLOCKS}
    model_cfg = OmegaConf.create(shared)

    # The `pretrain` block holds the base task HPs (loss, etc.); `finetune` is an
    # optional delta applied on top when fine-tuning (e.g. lowered loss weights,
    # FT lr). Either block may be absent — forecasters carry neither, and a model
    # that needs no FT tweaks simply omits `finetune` (base config is reused).
    if cfg.model.get("pretrain"):
        model_cfg = OmegaConf.merge(model_cfg, cfg.model.pretrain)
    if mode == "finetune" and cfg.model.get("finetune"):
        # `pretrained` (which checkpoint to init from) is stamped below in resolved form
        ft = {k: v for k, v in cfg.model.finetune.items() if k != "pretrained"}
        model_cfg = OmegaConf.merge(model_cfg, ft)

    # The mode determines the classification flag — never hand-set in configs.
    model_cfg.pretraining = (mode == "pretrain")

    # Stamp the resolved class/module so the saved config self-describes how to
    # rebuild the model (used by downstream fine-tuning).
    class_name, module = _class_and_module(cfg.model)
    model_cfg.model_name = class_name
    model_cfg.module = module
    if mode == "finetune":
        # ...and the weights it starts from ({run, epoch}; None = from scratch)
        model_cfg.pretrained = pretrained_source(cfg)

    # Runtime sizes are known only after the data is loaded.
    model_cfg.input_size = cfg.data_info.feature_size
    # output_size is only needed by the classifier head (fine-tuning); pretraining
    # datasets omit n_classes, so we leave it unset there.
    if cfg.data_info.get("n_classes") is not None:
        model_cfg.output_size = cfg.data_info.n_classes

    return model_cfg


def resolve_model(cfg):
    """Import and return the model class selected by ``cfg.model``."""
    model_name, module = _class_and_module(cfg.model)
    try:
        mod = import_module(f"src.models.{module}")
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"No module 'src.models.{module}' found for model '{model_name}'. "
            f"Set the config's 'module' field if the class lives in a "
            f"differently-named module."
        ) from e
    try:
        return getattr(mod, model_name)
    except AttributeError as e:
        raise AttributeError(
            f"'src.models.{module}' has no class '{model_name}'. "
            f"Is the class misnamed/not defined?"
        ) from e


def build_model(cfg):
    """
    Construct the model for ``cfg`` and return ``(model, model_cfg)``.

    In ``mode == finetune`` the model is initialised from ``model_cfg.pretrained``,
    the source resolved from ``cfg.model.finetune.pretrained``.
    """
    ModelClass = resolve_model(cfg)
    model_cfg = build_model_cfg(cfg)
    model = ModelClass(model_cfg)

    src = model_cfg.get("pretrained")
    if src:
        drop_keys = list(cfg.model.finetune.pretrained.get("drop_keys", ["clf"]))
        info = load_pretrained_state(model, src.run, src.epoch, drop_keys)
        model._pretrained_info = info  # for one-time logging by the caller
    return model, model_cfg


def resolve_dataset(cfg):
    """
    Load the dataset named by ``cfg.dataset.name``.

    Returns
    -------
    data : np.ndarray | torch.Tensor
        Main array of shape (N, T, C).
    extra_train_data : np.ndarray | torch.Tensor | None
        Optional extra data concatenated onto the training split only.
    """
    name = cfg.dataset.name
    try:
        module = import_module(f"src.datasets.{name}")
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"No module 'src.datasets.{name}' found. Check that the dataset name "
            f"matches its module name."
        ) from e
    try:
        loader = getattr(module, "load_pretraining_data")
    except AttributeError as e:
        raise AttributeError(
            f"'src.datasets.{name}' must define "
            f"'load_pretraining_data(ds_cfg) -> data' or '-> (data, extra_train_data)'."
        ) from e

    # The dataset config is passed so consolidated modules can dispatch on a
    # `variant` field (e.g. ukb_ica, ukb_aal). Simple datasets ignore it.
    result = loader(cfg.dataset)
    if isinstance(result, tuple):
        data, extra_train_data = result
    else:
        data, extra_train_data = result, None
    return data, extra_train_data


def resolve_finetuning_dataset(cfg):
    """Load labelled data for fine-tuning -> ``(data, labels)``."""
    name = cfg.dataset.name
    try:
        module = import_module(f"src.datasets.{name}")
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(f"No module 'src.datasets.{name}' found.") from e
    try:
        loader = getattr(module, "load_finetuning_data")
    except AttributeError as e:
        raise AttributeError(
            f"'src.datasets.{name}' must define "
            f"'load_finetuning_data(ds_cfg) -> (data, labels)' for fine-tuning."
        ) from e
    return loader(cfg.dataset)


def pretrained_source(cfg):
    """
    Weights fine-tuning starts from, ``{"run": <pretraining run dir>, "epoch": int}``,
    read from ``cfg.model.finetune.pretrained`` ("best" resolved via best_epoch.txt);
    None when that block is absent or has ``load: false`` (trained from scratch).
    """
    pre = (cfg.model.get("finetune") or {}).get("pretrained") or {}
    if not pre.get("run") or not pre.get("load", True):
        return None
    run = os.path.normpath(str(pre.run))
    epoch = pre.get("checkpoint", "best")
    if str(epoch) == "best":
        with open(os.path.join(run, "best_epoch.txt")) as f:
            epoch = f.read().strip()
    return {"run": run, "epoch": int(epoch)}


def load_pretrained_state(model, run_dir, epoch, drop_keys=("clf",)):
    """
    Load ``checkpoints/model_<epoch>.pt`` of a pretraining run into ``model``,
    strict except for state-dict keys containing one of ``drop_keys`` (e.g. the
    classifier head, which is trained from scratch).
    """
    import torch

    path = os.path.join(run_dir, "checkpoints", f"model_{int(epoch)}.pt")
    state = torch.load(path, map_location="cpu")
    pruned = {k: v for k, v in state.items()
              if not any(bad in k for bad in drop_keys)}
    missing, unexpected = model.load_state_dict(pruned, strict=False)

    # fail loudly unless only dropped keys (e.g. the fresh clf head) are missing
    bad_missing = [k for k in missing if not any(bad in k for bad in drop_keys)]
    if bad_missing or unexpected:
        raise ValueError(
            f"{path} does not match {type(model).__name__}: "
            f"missing={bad_missing[:5]}, unexpected={list(unexpected)[:5]}")
    return {"epoch": epoch, "loaded": len(pruned), "missing": list(missing)}
