from functools import reduce
from feature_engineering.pull_features.feat_demographics import feat_demographics
from feature_engineering.pull_features.feat_crime import feat_crime
from feature_engineering.pull_features.feat_prev_crime import feat_prev_crime
from feature_engineering.pull_features.feat_justice import feat_justice

def compose(*funcs):
  def _compose(f, g):
    # functions are expecting X, y not (X,y) so must unpack with *g
    return lambda *args, **kwargs: f(*g(*args, **kwargs))
  return reduce(_compose, funcs)

def feature_composition(*funcs):
  return compose(*reversed(funcs))

create_features = feature_composition(
  feat_demographics,
  feat_crime,
  feat_prev_crime,
  feat_justice
)
