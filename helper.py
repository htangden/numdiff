import numpy as np

def array_to_latex(arr, format_spec=".2f"):
  """Converts a 1D or 2D numpy array into a LaTeX bmatrix string.

  Parameters:
  - arr: numpy.ndarray (1D or 2D)
  - format_spec: string format specifier for numbers (default: '.2f')
  """
  arr = np.asarray(arr)

  if arr.ndim == 1:
    elements = [f"{x:{format_spec}}" for x in arr]
    return r"\begin{bmatrix} " + " & ".join(elements) + r" \end{bmatrix}"

  elif arr.ndim == 2:
    rows = []
    for row in arr:
      elements = [f"{x:{format_spec}}" for x in row]
      rows.append(" & ".join(elements))
    return r"\begin{bmatrix} " + r" \\ ".join(rows) + r" \end{bmatrix}"

  else:
    raise ValueError("Only 1D and 2D arrays are supported.")


