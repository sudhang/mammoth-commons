# Module Catalogue

This directory hosts a number of modules that are explicitly
supported by the MAMMOth-commons library's continuous integration.
Either create pull requests to add more modules here, 
or create new repositories for those modules.

:warning: If you need new data types to represent
the outcomes of loaders, these datatypes should be integrated
within the core of the MAMMOth-commons package. This way,
they can be imported from the `mammoth` module once
the package is installed within docker containers, which
in turn facilitates communication between loaders and 
metrics by referencing the same data types.

There are three subdirectories in which you can find
and place modules of respective types:

- `dataset_loaders/` contains components for loading datasets in various formats and internally converting them to some standard representations.
- `model_loaders/` contains components for loading various types of machine learning or other AI models.
- `metrics/` contains components that take as inputs a dataset loader and a model loader, perform some type of analysis using those, and output HTML or markdown.

All modules should be registered in the `dmonstrator/backend/loaders.py` and
`dmonstrator/backend/catalogue_loaders.py` files, 
at which point continuous integration adds them to the online
[catalogue website](https://mammoth-eu.github.io/mammoth-commons/).
For more instructions on how to contrivute new modules to this repository
[contribution guidelines](../CONTRIBUTING.md).


## Creating and deploying an external module

**Installation:** Install the latest version of `MAMMOth-commons`
and the `docker` package in your virtual environment:

*If you are working in your own repository:*

```bash
pip install --upgrade MAMMOth-commons
pip install docker
```

*If you are working in a clone of mammoth-commons and plan to create a pull request:*

```bash
pip install -e .
pip install requirements[test].txt # needed only if you want to run all integration tests
pip install docker
```

**New account:** You also need to create an account in
[DockerHub](!https://hub.docker.com/) or any other online
hosting service for docker images. You can ignore this step
while developing or testing modules.

**Required tools:** Finally, download, install, and run Decker Desktop
from [here](https://docs.docker.com/get-docker/). Command 
line instructions will use this to build docker images locally
before uploading them to the hosting service. You can also skip
this at the first stages of development.



Don't forget to set the correct module version first (if you reuse 
a previously uploaded version, the toolkit may not be able to see the change).
Then, [login to your docker account](https://docs.docker.com/engine/reference/commandline/login/).
For example, in the simplest case where you want to host your module
in DockerHub, it suffices to run the following command in your terminal:

```bash
docker login
```

This will ask for your DockerHub username (if you are not part of
a team in DockerHub, this should be the same as your namespace) 
and password. This way, your terminal will have
permission to push the created docker images there. 

Also make
sure that the library is visible to your virtual environment by calling
in the top level (from where you can access subdirecories 
*mammoth/*, *catalogue/*, *tests/*, etc)

```bash
pip install -e .
```


Finally, create and upload a module by running the following
command (kfp is installed alongside MAMMOth-commons):

```bash
kfp component build . --component-filepattern catalogue/metrics/model_card.py 
```

In this, replace the `test_modules/metric.py` with any other path
that contains the Python file in which you implemented your module. 

If you do *not* want to push the created docker image, for
example to run your new module in a local copy of the MAMMOth
bias toolkit without logging in and uploading it to DockerHub, run
this instead:

```bash
kfp component build . --component-filepattern catalogue/fairbench/modelcard.py --no-push-image
````

:warning: The build should be called from a directory where both your
module and virtual environment are subdirectories.



## For maintainers: continuous integration for module deployment

Use the [Toolkit Modules Build](https://github.com/mammoth-eu/mammoth-commons/actions/workflows/run-script.yml) 
action of this repository to make modules available for the MAI-BIAS toolkit server.

:warning: Use sparingly, as this consumes significant resources.

1. update the docker image version of all modules to the new one.
2. The module_yamls file is available. Download this and update it in the MAI-BIAS toolkit server. 
3. Folder named data yamls should be placed in components_yaml folder of toolkit.
4. Folder named meta yamls should be placed in components_metadata folder of toolkit.
