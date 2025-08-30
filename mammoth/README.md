#  MAMMOTH-commons: main package

This directory hosts the source code of the MAMMOth-commons package.
That is, import statements from the package, like 
`from mammoth.models import ONNX`, access the contents of this directory.

The current structure into subdirectories corresponding to Python modules is as follows:

| Subdirectory/File       | Description                                                                                                     |
|-------------------------|-----------------------------------------------------------------------------------------------------------------|
| `datasets/`             | Dataset **types** that are the outcomes of respective loader components and serve as inputs to metric components.|
| `models/`               | Model **types** that are the outcomes of respective loader components and serve as inputs to metric components.  |
| `integration.py`        | Implementation of decorators that create and actually decorate normal kfp methods.                              |
| `custom_kfp.py`         | Implementation of decorators that create and actually decorate normal kfp methods.                              |
| `testing.py`            | Implementation of functionality to strip away the decorators from the kfp integration process, and therefore allow unit and integration tests of developed components.|
| `externals.py`          | Supporting methods for component creation, for example that automate safe running of third-party code.          |

:warning: Only look and work with the `datasets/` and `models/` directories. The rest need some knowledge of the MAMMOth toolkit's internal workings to properly understand.
