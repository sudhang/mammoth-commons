import os

if os.getcwd().endswith(r"mammoth-commons\tests"):
    import warnings

    warnings.warn(
        "\nThere was an attempt to import mammoth-commons from within its `tests` folder."
        "\nThis could fail to import components from the `catalogue` folder for the tests,"
        "\nso `os.chdir('..')` command was applied first. If this still fails, make `import mammoth` "
        "\nyour first import. This message will not appear if you correctly set up a run configuration "
        "\nthat uses the top level of mammoth-commons as a working directory when running tests."
    )
    os.chdir("..")


def unwrap(component):
    return component.python_func.__mammoth_wrapped__


class Env:
    def __init__(self, *args):
        for v in args:
            v = unwrap(v)
            setattr(self, v.__name__, v)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
