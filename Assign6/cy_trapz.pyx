# cy_trapz.pyx
# Please run setup.py to increase the efficiency of this Cython script. If you have your own setup.py, please delete setup.py that is in my zip folder.

cimport cython  # Import Cython-specific declarations

# Disable bounds checking for array accesses to improve performance
@cython.boundscheck(False)
# Disable negative index wraparound to further enhance performance
@cython.wraparound(False)
cpdef cy_trapz(object f, double a, double b, long n):
    """
    Calculate the definite integral of function f from a to b using the trapezoidal rule with n trapezoids in Cython.

    Parameters:
    f (callable): The function to integrate. Must accept and return float.
    a (double): The lower limit of integration.
    b (double): The upper limit of integration.
    n (long): Number of trapezoids.

    Returns:
    double: Approximation of the integral.
    """
    cdef double h = (b - a) / n                 # Width of each trapezoid
    cdef double integral = 0.5 * (f(a) + f(b))  # Initialize integral with first and last terms
    cdef long i
    cdef double x
    for i in range(1, n):
        x = a + i * h
        integral += f(x)                        # Sum the function values at each interior point
    integral *= h                               # Multiply by the width to get the final integral
    return integral
