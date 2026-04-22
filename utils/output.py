import os


def resolve_output_dir(cfg):
    output_dir = getattr(cfg, "OUTPUT_DIR", "")
    if not output_dir:
        return ""

    path_parts = [output_dir]

    datasets_cfg = getattr(cfg, "DATASETS", None)
    exp_setting = getattr(datasets_cfg, "EXP_SETTING", "") if datasets_cfg is not None else ""
    if exp_setting:
        path_parts.append(exp_setting)

    run_name = getattr(cfg, "RUN_NAME", "")
    if run_name:
        path_parts.append(run_name)

    return os.path.join(*path_parts)
