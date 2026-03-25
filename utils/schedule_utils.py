import math


def _progress(task_index, total_tasks):
    if total_tasks is None or total_tasks <= 1:
        return 1.0
    return max(0.0, min(1.0, float(task_index - 1) / float(total_tasks - 1)))


def _interpolate(min_value, max_value, ratio):
    return min_value + (max_value - min_value) * ratio


def get_alpha_t(task_index, total_tasks, alpha_min=0.3, alpha_max=0.9, schedule='linear'):
    progress = _progress(task_index, total_tasks)
    if schedule == 'cosine':
        ratio = 0.5 * (1.0 - math.cos(math.pi * progress))
    elif schedule == 'exp':
        ratio = (math.exp(progress) - 1.0) / (math.e - 1.0)
    else:
        ratio = progress
    return _interpolate(alpha_min, alpha_max, ratio)


def get_rho_t(task_index, total_tasks, rho_min=1.0, rho_max=1.0, schedule='linear'):
    progress = _progress(task_index, total_tasks)
    if schedule == 'cosine':
        ratio = 0.5 * (1.0 - math.cos(math.pi * progress))
    elif schedule == 'exp':
        ratio = (math.exp(progress) - 1.0) / (math.e - 1.0)
    else:
        ratio = progress
    return _interpolate(rho_min, rho_max, ratio)
