# compare_methods.py

import timeit
import numpy as np
import math
from py_trapz import py_trapz  # Import pure Python trapezoidal implementation
import cy_trapz                # Import Cython trapezoidal implementation

# Define scalar functions for Python and Cython implementations
def f1_scalar(x):
    """Function f(x) = x^2."""
    return x ** 2

def f2_scalar(x):
    """Function f(x) = sin(x)."""
    return math.sin(x)

def f3_scalar(x):
    """Function f(x) = e^x."""
    return math.exp(x)

def f4_scalar(x):
    """Function f(x) = 1/x."""
    return 1 / x

# Define vectorized functions for NumPy implementation
def f1_vector(x):
    """Vectorized function f(x) = x^2 for NumPy."""
    return x ** 2  # NumPy handles array operations efficiently

def f2_vector(x):
    """Vectorized function f(x) = sin(x) for NumPy."""
    return np.sin(x)  # Use NumPy's sin for array inputs

def f3_vector(x):
    """Vectorized function f(x) = e^x for NumPy."""
    return np.exp(x)  # Use NumPy's exp for array inputs

def f4_vector(x):
    """Vectorized function f(x) = 1/x for NumPy."""
    return 1 / x  # NumPy handles division for arrays

# Configuration for timeit measurements
TIMEIT_REPEATS = 5    # Number of times to repeat the timing
TIMEIT_NUMBER = 1     # Number of executions per repeat

def test_case(f_scalar, f_vector, a, b, n, exact, description):
    """
    Executes and times the trapezoidal integration using Python, Cython, and NumPy implementations.
    
    Parameters:
    - f_scalar: Function for Python and Cython (scalar input)
    - f_vector: Function for NumPy (vectorized input)
    - a: Lower limit of integration
    - b: Upper limit of integration
    - n: Number of trapezoids
    - exact: Exact value of the integral for error calculation
    - description: Description of the test case
    """
    print(f"\nTest Case: {description}")
    print(f"Exact Integral: {exact}")

    # Python Trapezoidal Integration Timing
    timer_py = timeit.Timer(lambda: py_trapz(f_scalar, a, b, n))
    py_time = timer_py.timeit(number=TIMEIT_NUMBER)  # Measure execution time
    py_result = py_trapz(f_scalar, a, b, n)         # Get the result
    py_error = abs(py_result - exact)               # Calculate absolute error
    print(f"Python Trapezoidal: {py_result} (Error: {py_error}) in {py_time:.6f} seconds")

    # Cython Trapezoidal Integration Timing
    timer_cy = timeit.Timer(lambda: cy_trapz.cy_trapz(f_scalar, a, b, n))
    cy_time = timer_cy.timeit(number=TIMEIT_NUMBER)  # Measure execution time
    cy_result = cy_trapz.cy_trapz(f_scalar, a, b, n)  # Get the result
    cy_error = abs(cy_result - exact)                # Calculate absolute error
    print(f"Cython Trapezoidal: {cy_result} (Error: {cy_error}) in {cy_time:.6f} seconds")

    # NumPy Trapezoidal Integration Timing
    def numpy_trapz_callable():
        """Callable for NumPy trapezoidal integration."""
        x = np.linspace(a, b, n + 1)     # Generate evenly spaced sample points
        y = f_vector(x)                   # Compute function values
        return np.trapz(y, x)         # Perform integration

    timer_np = timeit.Timer(numpy_trapz_callable)
    np_time = timer_np.timeit(number=TIMEIT_NUMBER)  # Measure execution time
    x = np.linspace(a, b, n + 1)                      # Generate sample points
    y = f_vector(x)                                  # Compute function values
    np_result = np.trapz(y, x)                       # Perform integration
    np_error = abs(np_result - exact)                 # Calculate absolute error
    print(f"NumPy Trapezoidal: {np_result} (Error: {np_error}) in {np_time:.6f} seconds")

def main():
    """Main function to execute all test cases and the performance test."""
    # Define all test cases with their respective functions and exact integrals
    test_cases = [
        {
            "f_scalar": f1_scalar,
            "f_vector": f1_vector,
            "a": 0,
            "b": 1,
            "n": 1000,
            "exact": 1/3,
            "description": "f(x) = x^2 from 0 to 1"
        },
        {
            "f_scalar": f2_scalar,
            "f_vector": f2_vector,
            "a": 0,
            "b": math.pi,
            "n": 1000,
            "exact": 2.0,
            "description": "f(x) = sin(x) from 0 to π"
        },
        {
            "f_scalar": f3_scalar,
            "f_vector": f3_vector,
            "a": 0,
            "b": 1,
            "n": 1000,
            "exact": math.e - 1,
            "description": "f(x) = e^x from 0 to 1"
        },
        {
            "f_scalar": f4_scalar,
            "f_vector": f4_vector,
            "a": 1,
            "b": 2,
            "n": 1000,
            "exact": math.log(2),
            "description": "f(x) = 1/x from 1 to 2"
        },
    ]

    # Execute each test case
    for case in test_cases:
        test_case(
            f_scalar=case["f_scalar"],
            f_vector=case["f_vector"],
            a=case["a"],
            b=case["b"],
            n=case["n"],
            exact=case["exact"],
            description=case["description"]
        )

    # Performance Test Configuration
    print("\nPerformance Test: f(x) = x^2 from 0 to 10 with 10,000,000 trapezoids")
    f_perf_scalar = f1_scalar     # Scalar function for Python and Cython
    f_perf_vector = f1_vector     # Vectorized function for NumPy
    a_perf = 0                     # Lower limit
    b_perf = 10                    # Upper limit
    n_perf = 10_000_000            # Number of trapezoids
    exact_perf = (10 ** 3) / 3      # Exact integral value

    # Python Trapezoidal Integration Timing
    timer_py_perf = timeit.Timer(lambda: py_trapz(f_perf_scalar, a_perf, b_perf, n_perf))
    py_time = timer_py_perf.timeit(number=TIMEIT_NUMBER)  # Measure execution time
    py_result = py_trapz(f_perf_scalar, a_perf, b_perf, n_perf)  # Get the result
    py_error = abs(py_result - exact_perf)                     # Calculate absolute error
    print(f"Python Trapezoidal: {py_result} in {py_time:.6f} seconds")

    # Cython Trapezoidal Integration Timing
    timer_cy_perf = timeit.Timer(lambda: cy_trapz.cy_trapz(f_perf_scalar, a_perf, b_perf, n_perf))
    cy_time = timer_cy_perf.timeit(number=TIMEIT_NUMBER)  # Measure execution time
    cy_result = cy_trapz.cy_trapz(f_perf_scalar, a_perf, b_perf, n_perf)  # Get the result
    cy_error = abs(cy_result - exact_perf)                     # Calculate absolute error
    print(f"Cython Trapezoidal: {cy_result} in {cy_time:.6f} seconds")

    # NumPy Trapezoidal Integration Timing
    def numpy_trapz_perf():
        """Callable for NumPy trapezoidal integration in performance test."""
        x = np.linspace(a_perf, b_perf, n_perf + 1)  # Generate sample points
        y = f_perf_vector(x)                          # Compute function values
        return np.trapz(y, x)                    # Perform integration

    timer_np_perf = timeit.Timer(numpy_trapz_perf)
    np_time = timer_np_perf.timeit(number=TIMEIT_NUMBER)  # Measure execution time
    x = np.linspace(a_perf, b_perf, n_perf + 1)            # Generate sample points
    y = f_perf_vector(x)                                    # Compute function values
    np_result = np.trapz(y, x)                         # Perform integration
    np_error = abs(np_result - exact_perf)                  # Calculate absolute error
    print(f"NumPy Trapezoidal: {np_result} in {np_time:.6f} seconds")

if __name__ == "__main__":
    main()
