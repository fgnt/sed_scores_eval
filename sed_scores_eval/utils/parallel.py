import builtins
from functools import partial
from concurrent.futures import ProcessPoolExecutor


def _map_function_wrapper(args, arg_keys, func, **kwargs):
    if isinstance(arg_keys, (list, tuple)):
        assert len(args) == len(arg_keys), (len(args), len(arg_keys))
        return func(**{k: v for k, v in zip(arg_keys, args)}, **kwargs)
    assert isinstance(arg_keys, str), type(arg_keys)
    return func(**{arg_keys: args}, **kwargs)


def map(args, arg_keys, func, max_jobs, **kwargs):
    map_fn = partial(_map_function_wrapper, arg_keys=arg_keys, func=func, **kwargs)

    if isinstance(arg_keys, (list, tuple)):
        assert len(args) == len(arg_keys), (len(args), len(arg_keys))
        lengths = {len(arg) for arg in args if arg is not None}
        assert len(lengths) == 1, "all non-None args must have the same length."
        length = list(lengths)[0]
        args = list(zip(*[length * [None] if arg is None else arg for arg in args]))
        num_jobs = min(max_jobs, length)
    else:
        num_jobs = min(max_jobs, len(args))

    if num_jobs > 1:
        with ProcessPoolExecutor(num_jobs) as ex:
            return list(ex.map(map_fn, args))
    else:
        return list(builtins.map(map_fn, args))
