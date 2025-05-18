"""protect yourself from yourself"""

import functools
import typing
import numpy as np
import pandas as pd
import pandas.core.common as com


def add_rule(
    rules: typing.Dict,
    name: str,
    offset: float | tuple[float, float] = 0.0,
    scale: float | tuple[float, float] = 1.0,
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
    rules[name] = lambda x: np.add(np.multiply(x, scale), offset)
    return rules


def generate_rules(
    specification: typing.Any,
    offset: float | tuple[float, float] = 0.0,
    scale: float | tuple[float, float] = 1.0,
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


def norm_rules(df: pd.DataFrame, *columns: str) -> typing.Dict:
    """Generate transform rules to normalize data for a mean
    value of zero and a standard deviation of one.

    Parameters
    ==========
    df :: pandas.DataFrame
    column(s) :: str

    Returns
    =======
    rule :: dict(name=Function)
    """
    specification = []
    for name in columns:
        data = df[name]
        scale = 1.0 / data.std()
        offset = -(np.multiply(data, scale)).mean()
        specification.append((name, offset, scale))
    rules = generate_rules(specification=specification)
    return rules


def normalize(df: pd.DataFrame, *columns: str) -> pd.DataFrame:
    """
    Transform data for a mean value of zero and a standard 
    deviation of one.
    
    Parameters
    ==========
    df :: pandas.DataFrame
    column(s) :: str

    Returns
    =======
    :: pandas.DataFrame
    """ ""
    rules = norm_rules(df, *columns)
    return blind(df, rules)


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


def blindat(
    func: typing.Callable | None = None,
    default_rules: typing.Dict | None = None,
    rules_keyword: str = "rules",
) -> typing.Callable:
    """Decorator to blind the results of functions or methods
    that return a pandas.DataFrame as the first or only result.

    Parameters
    ----------
    func :: Function
    default_rules :: dict (default=None)
    rules_keyword :: str (default="rules")

    Example
    =======

    @blindat
    def load(rules=None):
        return pandas.DataFrame(data)

    # ... which is equivalent to
    def load(**kwargs):
        rules = kwargs.pop("rules", None)
        return blind(pandas.DataFrame(data), rules)
    """
    if func is None:
        return functools.partial(
            blindat, default_rules=default_rules, rules_keyword=rules_keyword
        )

    @functools.wraps(func)
    def wrapper(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        rules = kwargs.pop(rules_keyword, default_rules)
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
