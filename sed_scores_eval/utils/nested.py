from typing import Optional


def flatten(d, sep: Optional[str] = '.', *, flat_type=dict):
    """
    Flatten a nested `dict` using a specific separator.

    !!! Copied from https://github.com/fgnt/paderbox/blob/master/paderbox/utils/nested.py !!!

    Args:
        sep: When `None`, return `dict` with `tuple` keys (guarantees inversion
                of flatten) else join the keys with sep
        flat_type:  Allow other mappings instead of `flat_type` to be
                flattened, e.g. using an isinstance check.

    import collections
    flat_type=collections.abc.MutableMapping

    >>> d_in = {'a': 1, 'c': {'a': 2, 'b': {'x': 5, 'y' : 10}}, 'd': [1, 2, 3]}
    >>> d = flatten(d_in)
    >>> for k, v in d.items(): print(k, v)
    a 1
    c.a 2
    c.b.x 5
    c.b.y 10
    d [1, 2, 3]
    >>> d = flatten(d_in, sep='_')
    >>> for k, v in d.items(): print(k, v)
    a 1
    c_a 2
    c_b_x 5
    c_b_y 10
    d [1, 2, 3]
    """

    # https://stackoverflow.com/a/6027615/5766934

    # {k: v for k, v in d.items()}

    def inner(d, parent_key):
        items = {}
        for k, v in d.items():
            new_key = parent_key + (k,)
            if isinstance(v, flat_type) and v:
                items.update(inner(v, new_key))
            else:
                items[new_key] = v
        return items

    items = inner(d, ())
    if sep is None:
        return items
    else:
        return {
            sep.join(k): v for k, v in items.items()
        }


def deflatten(d: dict, sep: Optional[str] = '.', maxdepth: int = -1):
    """
    Build a nested `dict` from a flat dict respecting a separator.

    !!! Copied from https://github.com/fgnt/paderbox/blob/master/paderbox/utils/nested.py !!!

    Args:
        d: Flattened `dict` to reconstruct a `nested` dict from
        sep: The separator used in the keys of `d`. If `None`, `d.keys()` should
            only contain `tuple`s.
        maxdepth: Maximum depth until wich nested conversion is performed

    >>> d_in = {'a': 1, 'c': {'a': 2, 'b': {'x': 5, 'y' : 10}}, 'd': [1, 2, 3]}
    >>> d = flatten(d_in)
    >>> for k, v in d.items(): print(k, v)
    a 1
    c.a 2
    c.b.x 5
    c.b.y 10
    d [1, 2, 3]
    >>> deflatten(d)
    {'a': 1, 'c': {'a': 2, 'b': {'x': 5, 'y': 10}}, 'd': [1, 2, 3]}
    >>> deflatten(d, maxdepth=1)
    {'a': 1, 'c': {'a': 2, 'b.x': 5, 'b.y': 10}, 'd': [1, 2, 3]}
    >>> deflatten(d, maxdepth=0)
    {'a': 1, 'c.a': 2, 'c.b.x': 5, 'c.b.y': 10, 'd': [1, 2, 3]}
    >>> d = flatten(d_in, sep='_')
    >>> for k, v in d.items(): print(k, v)
    a 1
    c_a 2
    c_b_x 5
    c_b_y 10
    d [1, 2, 3]
    >>> deflatten(d, sep='_')
    {'a': 1, 'c': {'a': 2, 'b': {'x': 5, 'y': 10}}, 'd': [1, 2, 3]}
    >>> deflatten({('a', 'b'): 'd', ('a', 'c'): 'e'}, sep=None)
    {'a': {'b': 'd', 'c': 'e'}}
    >>> deflatten({'a.b': 1, 'a': 2})
    Traceback (most recent call last):
      ...
    AssertionError: Conflicting keys! ('a',)
    >>> deflatten({'a': 1, 'a.b': 2})
    Traceback (most recent call last):
      ...
    AssertionError: Conflicting keys! ('a', 'b')

    """
    ret = {}
    if sep is not None:
        d = {
            tuple(k.split(sep, maxdepth)): v for k, v in d.items()
        }

    for keys, v in d.items():
        sub_dict = ret
        for sub_key in keys[:-1]:
            if sub_key not in sub_dict:
                sub_dict[sub_key] = {}
            assert isinstance(sub_dict[sub_key], dict), (
                f'Conflicting keys! {keys}'
            )
            sub_dict = sub_dict[sub_key]
        assert keys[-1] not in sub_dict, f'Conflicting keys! {keys}'
        sub_dict[keys[-1]] = v
    return ret
