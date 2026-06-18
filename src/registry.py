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
A model class ``Foo`` is imported from ``src.models.<module>`` where ``module``
defaults to the class name (``cfg.model.module`` overrides it, so several classes
can share one module file, e.g. the DECIFRA_* variants in ``src.models.DECIFRA``).
HPs are canonical in the yaml; ``default_HPs``/``custom_HPs`` are no longer used.

A model config carries shared architecture keys plus optional task blocks::

    model_name: DECIFRA_MS
    module: DECIFRA_MS
    rnn: {...}                 # shared architecture
    pretrain:  {pretraining: true,  loss: {...}}
    finetune:  {pretraining: false, loss: {...}, load_pretrained: true}

``build_model_cfg`` flattens the block named by ``cfg.mode`` onto the shared keys.

Datasets
--------
A dataset ``bar`` is resolved from ``src.datasets.bar`` and must expose
``load_pretraining_data() -> data`` or ``-> (data, extra_train_data)``.
"""

from importlib import import_module

from omegaconf import OmegaConf


# Keys that are task blocks rather than shared model params.
_TASK_BLOCKS = ("pretrain", "finetune")


def build_model_cfg(cfg):
    """
    Assemble the effective, flat model config the model class consumes.

    Flattens the task block named by ``cfg.mode`` onto the shared model params,
    drops the unused blocks, and injects runtime sizes from ``cfg.data_info``.
    """
    mode = cfg.mode
    shared = {k: v for k, v in cfg.model.items() if k not in _TASK_BLOCKS}
    block = cfg.model.get(mode, {}) or {}
    model_cfg = OmegaConf.merge(OmegaConf.create(shared), block)

    # Runtime sizes are known only after the data is loaded.
    model_cfg.input_size = cfg.data_info.feature_size
    # output_size is only needed by the classifier head (fine-tuning); pretraining
    # datasets omit n_classes, so we leave it unset there.
    if cfg.data_info.get("n_classes") is not None:
        model_cfg.output_size = cfg.data_info.n_classes

    return model_cfg


def resolve_model(cfg):
    """Import and return the model class named by ``cfg.model.model_name``."""
    model_name = cfg.model.model_name
    module = cfg.model.get("module") or model_name
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
