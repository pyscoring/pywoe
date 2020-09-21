
![image](https://avatars2.githubusercontent.com/u/71639999?s=200&v=4)

# **pywoe** [Beta]

The missing *scikit-learn* addition to work with Weight-of-Evidence scoring, 
with a special focus on credit risk modelling. There's evidently a lack of
open source, free-to-use, well-tested Python package for basic credit risk
modelling tasks. Such a package should provide easily serialisable, deployable,
transferable data validation, feature engineering and feature selection techniques.
It should also be easy to use within the Jupyter Lab framework.

This is still very much a work-in-progress, and the package can be extended in 
multiple useful ways. Feel free to contribute.

# Table of Contents

1. [Installation](#installation)
2. [Usage Examples](#basic-examples)
3. [Further Work](#further)

<a name="installation"></a>
## Installation

To install the latest version of the package, simply run

```bash
pip install pywoe
```

<a name="basic-examples"></a>
## Usage Examples

### Introduction

For easy start, there's a ready-made `sklearn` pipeline provided.
To load, do the following. Feel free to run the pipeline on example data, 
as below.

```python
from pywoe.interface import get_raw_data_to_woe_values_pipeline
from sklearn.datasets import load_breast_cancer

pipeline = get_raw_data_to_woe_values_pipeline()
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
pipeline.fit(X, y)
woe_transformed = pipeline.transform(X)
```

The setup above automatically constructs bins and computes WoE across them.
The output can be used to select features for a logistic regression model,
or to preprocess features before entering them to a model.

### Informaton Values (IV)

In the example above, Information Values have also been computed. To retrieve 
them alongside the binning decided for a feature `mean radius`, do:

```python
pipeline['woe_transformer'].woe_spec['mean radius'].bins
```

and you'll see the values printed out.

### Inspecting Default Settings

```python
from pywoe import constants

constants.NUMERIC_ACCURACY
constants.DEFAULT_DECISION_TREE_CLASSIFIER_FIT_KWARGS
constants.DEFAULT_DECISION_TREE_CLASSIFIER_INIT_KWARGS
constants.P_VALUE_THRESHOLD
```

### Overriding Defaults

```python
from sklearn.pipeline import Pipeline
from pywoe.feature_engineering.validator import FeatureValidator
from pywoe.feature_engineering.binning import DecisionTreeBinner
from pywoe.feature_engineering.woe import WoETransformer

feature_validator = FeatureValidator()
binner = DecisionTreeBinner(
    feature_validator=feature_validator,
    init_kwargs={
        "criterion": "entropy",
        "max_depth": 3,
        "min_samples_leaf": 0.2
    }
)
woe_transformer = WoETransformer(binner=binner)

# Keep in mind `binner` is not an `sklearn` object, it is a parameter 
# to `woe_transformer`, so it's not used in the pipeline.
pipeline = Pipeline([
    ('validator', feature_validator),
    ('woe_transformer', woe_transformer)
])
```

<a name="further"></a>
## Further Work

Further work needed includes, but is not limited to:
 * (significantly) improving testing,
 * adding marginal-IV-based automated feature selection,
 * adding Jupyter-integrated plotting capabilities to inspect models,
 * adding residual monitoring (ReMo) capabilities,
 * ...