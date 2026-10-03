"""
Optional plugin configuration for feature extraction.

You can register additional datasets / reps by:

1. Creating a new Python module, e.g. src/feature_extraction/my_new_rep.py
   that imports `register_dataset_handler` / `register_rep_handler` from
   `registry.py` and calls them at import time.
2. Adding the module's dotted path to the lists below.

The main CLI in process_datasets.py will try to import every module listed
here once at startup, so you don't need to touch the core script to extend
functionality.
"""

# Dotted module paths that define extra DatasetHandlers / RepHandlers.
EXTRA_DATASET_MODULES = [
    # "src.feature_extraction.my_new_dataset",
]

EXTRA_REP_MODULES = [
    # "src.feature_extraction.my_new_rep",
]

