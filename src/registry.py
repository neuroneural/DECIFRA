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
    finetune:  {pretraining: false, loss: {...}, load_pretrained: true}

``build_model_cfg`` flattens the block named by ``cfg.mode`` onto the shared keys
and stamps the resolved ``model_name``/``module`` so the saved config self-describes.

Datasets
--------
A dataset ``bar`` is resolved from ``src.datasets.bar`` and must expose
``load_pretraining_data() -> data`` or ``-> (data, extra_train_data)``.
"""

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
        model_cfg = OmegaConf.merge(model_cfg, cfg.model.finetune)

    # The mode determines the classification flag — never hand-set in configs.
    model_cfg.pretraining = (mode == "pretrain")

    # Stamp the resolved class/module so the saved config self-describes how to
    # rebuild the model (used by downstream fine-tuning).
    class_name, module = _class_and_module(cfg.model)
    model_cfg.model_name = class_name
    model_cfg.module = module

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


def load_pretrained_state(model, run_dir, checkpoint="best", drop_keys=("clf",)):
    """
    Load weights from a pretraining run into ``model`` (strict=False), skipping
    any state-dict keys containing one of ``drop_keys`` (e.g. the classifier head,
    which is trained from scratch). ``checkpoint`` is "best" (resolved via
    best_epoch.txt) or an integer epoch.
    """
    import os
    import torch

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if str(checkpoint) == "best":
        with open(os.path.join(run_dir, "best_epoch.txt")) as f:
            epoch = int(f.read().strip())
    else:
        epoch = int(checkpoint)
    path = os.path.join(ckpt_dir, f"model_{epoch}.pt")
    state = torch.load(path, map_location="cpu")
    pruned = {k: v for k, v in state.items()
              if not any(bad in k for bad in drop_keys)}
    missing, unexpected = model.load_state_dict(pruned, strict=False)
    return {"epoch": epoch, "loaded": len(pruned),
            "missing": list(missing), "unexpected": list(unexpected)}
