from typing import Callable

def compose(*functions):
    # TODO: implement with reduce
    # TODO: implement lazy load
    if not functions:
        raise ValueError("'functions' must not be empty")
    if len(functions) == 1:
        return lambda *args, **kwargs: functions[0](*args, **kwargs)
    return lambda *args, **kwargs: compose(*functions[:-1])(functions[-1](*args, **kwargs))

def fluent_compose(*functions):
    if not functions:
        raise ValueError("'functions' must not be empty")
    def _exec(*args, **kwargs):
        y = functions[0](*args, **kwargs)
        for function in functions[1:]:
            y = function(y)
        return y
    return _exec
    #return compose(*reversed(functions))

if __name__ == '__main__':
    f = compose(
        lambda x: x + 1,
        lambda x: x * 2,
        lambda x: x + 3,
        lambda p: p * 3
    )
    print(f(p=7))