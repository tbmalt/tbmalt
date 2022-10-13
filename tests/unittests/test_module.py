
from tbmalt.ml.module import Calculator, require_args, requires_args, \
    gather_args_from, call_with_required_args


def func_1(x):
    return x == 1


def func_2(x=None):
    return x == 1


@require_args("arg_x")
def func_3(x):
    return x == 1


@require_args("arg_x", "arg_y")
def func_4(x, y):
    return x == 1 and y == 2


@require_args(x="kwarg_x")
def func_5(x=None):
    return x == 10


@require_args(x="kwarg_x", y="kwarg_y")
def func_6(x=None, y=None):
    return x == 10 and y == 20


@require_args("arg_x", y="kwarg_y")
def func_7(x=None, y=None):
    return x == 1 and y == 20


@require_args("arg_x", "arg_y")
def func_8(x=None, y=None):
    return x == 1 and y == 2


class DummyClass:
    def __init__(self, dummy_feed_function):
        self.dummy_feed = dummy_feed_function
        self.arg_x = 1
        self.arg_y = 2
        self.kwarg_x = 10
        self.kwarg_y = 20

    def __call__(self):
        if requires_args(self.dummy_feed):
            return call_with_required_args(self.dummy_feed, self)
        else:
            # Ensure the code can default some hard coded argument set when no
            # required arguments are provided.
            return self.dummy_feed(self.arg_x)


def test_gather_args_from():
    # Ensure that the `gather_args_from` function operates as expected.
    funcs = [func_1, func_2, func_3, func_4, func_5, func_6, func_7, func_8]
    for func in funcs:
        assert DummyClass(func)()
