import inspect
from typing import get_type_hints
import markdown2


def format_name(name):
    ret = " ".join(name.split("_")).replace("data ", "").replace("model ", "")
    ret = ret[0].upper() + ret[1:]
    return ret


class Registry:
    def __init__(self, desktopmode=True):
        self.dataset_loaders = dict()
        self.model_loaders = dict()
        self.analysis_methods = dict()
        self.name_to_runnable = dict()
        self.parameters_to_class = dict()
        self.desktopmode = desktopmode

    def data(self, component, compatible=None):
        self._register(self.dataset_loaders, component, compatible)

    def model(self, component, compatible=None):
        self._register(self.model_loaders, component, compatible)

    def analysis(self, component, compatible=None):
        self._register(self.analysis_methods, component, compatible)

    def _register(self, catalogue: dict, component, compatible=None):
        component = component.python_func.__mammoth_wrapped__
        signature = inspect.signature(component)
        type_hints = get_type_hints(component)

        # find argument descriptions
        doc = ""
        args_desc = dict()
        args_options = dict()
        started_args = False
        separator_title = " "
        sep_title = separator_title
        started_options = False
        for line in component.__doc__.split("\n"):
            line = line.strip()
            if line.startswith("Options:"):
                started_options = True
            elif line.startswith("Args:"):
                started_args = True
            elif line.endswith(" args:"):
                separator_title = line[:-5].strip()
                sep_title = separator_title
                if separator_title:
                    separator_title = "<br><h3>" + separator_title + "</h3>"
            elif started_options and ":" in line:
                splt = line.split(":", maxsplit=2)
                args_options[splt[0]] = [
                    option.strip() for option in splt[1].split(",")
                ]
            elif started_args and ":" in line:
                splt = line.split(":", maxsplit=2)
                name = format_name(splt[0]).replace(sep_title + " ", "")
                name = name[0].upper() + name[1:]
                # args_desc[splt[0]] = f"{separator_title}<i>{name} - </i> {splt[1]}"

                args_desc[splt[0]] = (
                    f"""<h1>{separator_title} {name}</h1> {splt[1]}"""
                    if self.desktopmode
                    else f"""<button
                          type="button"
                          class="btn btn-light"
                          data-bs-toggle="tooltip"
                          data-bs-placement="top"
                          title="{splt[1]}"
                          data-description="{splt[1]}"
                          data-name="{name}"
                          onclick="showDescriptionModal(this)">
                          <i class="bi bi-info-circle"></i> {name}
                        </button>"""
                )

                separator_title = ""
            else:
                doc += line + "\n"

        args = list()
        args_to_classes = dict()
        for pname, parameter in signature.parameters.items():
            arg_type = type_hints.get(pname, parameter.annotation)
            assert pname != "return"
            args_to_classes[pname] = arg_type
            arg_type = arg_type.__name__
            if arg_type == "str" and (
                "path" in pname.lower() or "url" in pname.lower()
            ):
                arg_type = "url"
            if parameter.default is not inspect.Parameter.empty:  # ignore kwargs
                args.append(
                    [
                        pname,
                        arg_type,
                        "None" if parameter.default is None else parameter.default,
                        args_desc.get(pname, format_name(pname)),
                    ]
                )
            else:
                args.append(
                    [pname, arg_type, "None", args_desc.get(pname, format_name(pname))]
                )

        name = format_name(component.__name__)
        assert name not in self.name_to_runnable
        self.name_to_runnable[name] = component
        catalogue[name] = {
            "description": str(
                markdown2.markdown(
                    "\n".join([line for line in doc.replace("_", " ").split("\n")]),
                    extras=["tables", "fenced-code-blocks", "code-friendly"],
                )
            ).replace("\n", " "),
            "parameters": args,
            "parameter_options": args_options,
            "name": component.__name__,
            "compatible": (
                []
                if compatible is None
                else [
                    format_name(c.python_func.__mammoth_wrapped__.__name__)
                    for c in compatible
                ]
            ),
            "return": signature.return_annotation.__name__,
        }
        args_to_classes["return"] = signature.return_annotation
        self.parameters_to_class[name] = args_to_classes
