import functools
import typing
import numpy as np
import pandas as pd
import pandas.core.common as com


def add_rule(
    rules: typing.Dict,
    name: str,
    offset: int | float = 0.0,
    scale: int | float = 1.0,
    verbose: bool = False,
) -> typing.Dict:
    """Add a new rule to a dictionary of rules.

    If offset or scale are given as `(start, stop)`,
    a random value is selected from within the range.

    Parameters
    ==========
    rules :: dict(name=Function)
    name :: str
    offset=0.0 :: float or (start::float, stop::float)
    scale=1.0 :: float or (start::float, stop::float)
    verbose=False :: bool

    Returns
    =======
    :: dict(name=Function)
    """
    # testing parameters - NOT CURRENTLY USED!
    assign = False  ## format functions for pd.DataFrame.assign instead of pd.DataFrame.transform.
    preffix = "blind_"  ## preffix column names
    # __main__
    if verbose:
        print(f"Data transformation parameters for column = {name}:")
    if isinstance(offset, tuple):
        # generate random offset within limits
        _start, _stop = offset
        offset = _start + np.random.rand() * (_stop - _start)
    if verbose:
        print(f"    {name}.offset = {offset}")
    if isinstance(scale, tuple):
        # generate random scale within limits
        _start, _stop = scale
        scale = _start + np.random.rand() * (_stop - _start)
    if verbose:
        print(f"    {name}.scale = {scale}")
    if assign:
        # pd.DataFrame.assign(**rules)
        rules[preffix + name] = lambda x: np.add(np.multiply(x[name], scale), offset)
    else:
        # pd.DataFrame.apply(rules) or pd.DataFrame.transform(rules)
        rules[name] = lambda x: np.add(np.multiply(x, scale), offset)
    return rules


def generate_rules(
    specification: typing.Any,
    offset: int | float = 0.0,
    scale: int | float = 1.0,
    random_seed: int | None = None,
    verbose: bool = False,
) -> typing.Dict:
    """Generate a dictionary of linear transform rules.

    If offset or scale are given as two-valued tuples,
    a random values is selected from within the range.

    Parameters
    ==========
    specification : column name :: str
                    column names :: list(str)
                    :: list(tuple(name, offset, scale))
                    :: dict(name=dict(offset=offset, scale=scale))
    offset=0.0 :: float or (start::float, stop::float)
    scale=1.0 :: float or (start::float, stop::float)
    random_seed=None :: int
    verbose=False :: bool

    Returns
    =======
    :: dict(name=Function)

    Examples
    ========
    specification = [("A", (10.0, 20.0), (0.9, 1.1)), ("B", 0.5, 1.2)]

    specification = {
        "A": {"offset": (10.0, 20.0), "scale": (0.9, 1.1)},
        "B": {"offset": 0.5, "scale": 1.2},
    }
    """
    rules: typing.Dict = dict()
    np.random.seed(random_seed)
    match specification:
        case str(name):
            rules = add_rule(rules, name, offset=offset, scale=scale, verbose=verbose)
        case [str(_name), *_]:
            for name in specification:
                rules = add_rule(
                    rules, name, offset=offset, scale=scale, verbose=verbose
                )
        case [(str(_name), _offset, _scale), *_]:
            for name, offset, scale in specification:
                rules = add_rule(
                    rules, name, offset=offset, scale=scale, verbose=verbose
                )
        case {**_items}:
            for name, spec in specification.items():
                rules = add_rule(
                    rules,
                    name,
                    offset=spec.get("offset", offset),
                    scale=spec.get("scale", scale),
                    verbose=verbose,
                )
        case _:
            raise TypeError("invalid rule specification")
    return rules


def blind(df: pd.DataFrame, rules: typing.Dict) -> pd.DataFrame:
    """Simular functionality to pd.DataFrame.transform(rules),
    except columns without rules are not dropped.

    Parameters
    ==========
    df :: pandas.DataFrame
    rules :: dict(name=Function)

    Returns
    =======
    :: pandas.DataFrame
    """
    df = df.copy(deep=None)
    for k, v in rules.items():
        # df[k] = df[k].transform(v)             # slow AF
        df[k] = com.apply_if_callable(v, df[k])  # why so fast?!
    return df


def inspect(rules: typing.Dict) -> typing.Dict:
    """Extract offset and scale values from a dictionary
    of linear functions.

    Parameters
    ==========
    rules :: dict(name=Function)

    Returns
    =======
    :: dict(name=dict(offset=float, scale=float))
    """
    return {
        name: {"offset": func(0.0), "scale": func(1.0) - func(0.0)}
        for name, func in rules.items()
    }


def obfuscate(
    func: typing.Callable | None = None, default_rules: typing.Dict | None = None
) -> typing.Callable:
    """Decorator to blind the results of functions or methods
    that return a pandas.DataFrame as the first or only result.

    Parameters
    ----------
    func :: Function
    default_rules :: dict (default=None)

    Example
    =======

    @obfuscate
    def load(transform=None):
        return pandas.DataFrame(data)

    # ... which is equivalent to
    def load(**kwargs):
        rules = kwargs.pop("transform", None)
        return blind(pandas.DataFrame(data), rules=rules)
    """
    if func is None:
        return functools.partial(obfuscate, default_rules=default_rules)

    @functools.wraps(func)
    def wrapper(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        rules = kwargs.pop("rules", default_rules)
        result = func(*args, **kwargs)
        # no transform
        if rules is None:
            return result
        # blind data
        assert isinstance(rules, dict), "invalid rules"
        match result:
            case pd.DataFrame():
                # DataFrame
                return blind(result, rules)
            case pd.DataFrame(), *rest:
                # first element is DataFrame
                return blind(result[0], rules), *rest
            case _:
                TypeError("func result does not match a valid case")

    return wrapper
