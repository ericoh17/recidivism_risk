from functools import reduce
from feature_engineering.pull_features.feat_init import feat_init

def compose(*funcs):
  def _compose(f, g):
    # functions are expecting X, y not (X,y) so must unpack with *g
    return lambda *args, **kwargs: f(*g(*args, **kwargs))
  return reduce(_compose, funcs)

def feature_composition(*funcs):
  return compose(*reversed(funcs))

create_features = feature_composition(
  feat_init
)
