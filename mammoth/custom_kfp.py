import inspect
import pathlib
from typing import Callable, List, Optional
import warnings
from kfp.dsl import python_component
from kfp.dsl import structures
from kfp.dsl.component_factory import (
    extract_component_interface,
    REGISTERED_MODULES,
    _python_function_name_to_component_name,
    ComponentInfo,
    _get_packages_to_install_command,
    _DEFAULT_BASE_IMAGE,
    _get_command_and_args_for_containerized_component,
    _get_command_and_args_for_lightweight_component,
)


def custom_create_component_from_func(
    func: Callable,
    true_func: Optional[str] = None,
    base_image: Optional[str] = None,
    target_image: Optional[str] = None,
    packages_to_install: List[str] = None,
    pip_index_urls: Optional[List[str]] = None,
    output_component_file: Optional[str] = None,
    install_kfp_package: bool = True,
    kfp_package_path: Optional[str] = None,
) -> python_component.PythonComponent:
    """Implementation for the @component decorator.

    The decorator is defined under component_decorator.py. See the
    decorator for the canonical documentation for this function.
    """

    packages_to_install_command = _get_packages_to_install_command(
        install_kfp_package=install_kfp_package,
        target_image=target_image,
        kfp_package_path=kfp_package_path,
        packages_to_install=packages_to_install,
        pip_index_urls=pip_index_urls,
    )

    command = []
    args = []
    if base_image is None:
        base_image = _DEFAULT_BASE_IMAGE
        warnings.warn(
            (
                "Python 3.7 has reached end-of-life. The default base_image used by the @dsl.component decorator will switch from 'python:3.7' to 'python:3.8' on April 23, 2024. To ensure your existing components work with versions of the KFP SDK released after that date, you should provide an explicit base_image argument and ensure your component works as intended on Python 3.8."
            ),
            FutureWarning,
            stacklevel=2,
        )

    component_image = base_image

    if target_image:
        component_image = target_image
        command, args = _get_command_and_args_for_containerized_component(
            function_name=true_func.__name__,
        )
    else:
        command, args = _get_command_and_args_for_lightweight_component(func=func)

    component_spec = extract_component_interface(func)
    component_spec.implementation = structures.Implementation(
        container=structures.ContainerSpecImplementation(
            image=component_image,
            command=packages_to_install_command + command,
            args=args,
        )
    )

    module_path = pathlib.Path(inspect.getsourcefile(true_func))
    module_path.resolve()

    component_name = _python_function_name_to_component_name(true_func.__name__)
    component_info = ComponentInfo(
        name=component_name,
        function_name=true_func.__name__,
        func=func,
        target_image=target_image,
        module_path=module_path,
        component_spec=component_spec,
        output_component_file=output_component_file,
        base_image=base_image,
        packages_to_install=packages_to_install,
        pip_index_urls=pip_index_urls,
    )

    if REGISTERED_MODULES is not None:
        REGISTERED_MODULES[component_name] = component_info

    if output_component_file:
        component_spec.save_to_component_yaml(output_component_file)

    return python_component.PythonComponent(
        component_spec=component_spec, python_func=func
    )
