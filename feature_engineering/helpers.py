# flatten values of nested sequences
def flatten(l):
  for el in l:
    try:
      yield from flatten(el)
    except TypeError:
      yield el

