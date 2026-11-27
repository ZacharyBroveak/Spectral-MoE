# Spectral-MoE: src/evaluation/revisions.py
# Author: Zachary Broveak
#
# Pinned dataset revisions for evaluation.
#
# lm-eval loads each task's dataset from the Hugging Face Hub by name, which would silently
# follow upstream changes. task_configs() loads lm-eval's own YAML definition of each task,
# verifies that it still points at the expected dataset, and adds the pinned revision to its
# dataset_kwargs. Everything else (prompts, splits, metrics) stays as defined by the lm-eval
# version pinned in requirements.txt. If a future harness moves a task to another dataset,
# this fails loudly instead of measuring something different.
#
# WIKITEXT_REVISION pins the perplexity corpus and the router refit diagnostics.

WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
TASK_REVISIONS = {
    "boolq": ("aps/super_glue", "3de24cf8022e94f4ee4b9d55a6f539891524d646"),
    "sciq": ("allenai/sciq", "2c94ad3e1aafab77146f384e23536f97a4849815"),
    "piqa": ("baber/piqa", "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1"),
    "winogrande": ("allenai/winogrande", "01e74176c63542e6b0bcb004dcdea22d94fb67b5"),
    "arc_challenge": ("allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453"),
    "hellaswag": ("Rowan/hellaswag", "218ec52e09a7e7462a5400043bb9a69a41d06b76"),
}


def task_configs(names):
    from lm_eval.tasks import TaskManager
    from lm_eval.tasks._yaml_loader import load_yaml
    manager = TaskManager()
    configs = []
    for name in names:
        cfg = load_yaml(manager.task_index[name].yaml_path, resolve_func=True)
        dataset, revision = TASK_REVISIONS[name]
        if cfg['dataset_path'] != dataset:
            raise ValueError(f"Harness task {name} changed its dataset")
        # Pin the revision without discarding other dataset options.
        cfg['dataset_kwargs'] = {**(cfg.get('dataset_kwargs') or {}), 'revision': revision}
        configs.append(cfg)
    return configs
