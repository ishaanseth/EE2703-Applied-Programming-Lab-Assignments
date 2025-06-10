# py_trapz.py

def py_trapz(f, a, b, n):
    """
    Calculate the definite integral of function f from a to b using the trapezoidal rule with n trapezoids.

    Parameters:
    f (callable): The function to integrate.
    a (float): The lower limit of integration.
    b (float): The upper limit of integration.
    n (int): Number of trapezoids.

    Returns:
    float: Approximation of the integral.
    """
    h = (b - a) / n                 # Width of each trapezoid
    integral = 0.5 * (f(a) + f(b))  # Initialize integral with first and last terms
    for i in range(1, n):
        x = a + i * h
        integral += f(x)            # Sum the function values at each interior point
    integral *= h                   # Multiply by the width to get the final integral
    return integral
