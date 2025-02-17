import time
import matplotlib.pyplot as plt
import numpy as np
from functools import lru_cache
import sys
sys.setrecursionlimit(10**6)


test_indexes = (5, 7, 10, 12, 15, 17, 20, 22, 25, 27, 30, 32, 35, 37, 40, 42, 45)
compare_indexes = (501, 631, 794, 1000, 1259, 1585, 1995, 2512, 3162, 3981, 5012, 6310, 7943, 10000, 12589, 15849)

# FAST EXPONENTIAL
class M2x2:
    def __init__(self, e00, e01, e10, e11):
        self.e00 = e00
        self.e01 = e01
        self.e10 = e10
        self.e11 = e11

    def __mul__(self, other):
        return M2x2(
            self.e00 * other.e00 + self.e01 * other.e10,
            self.e00 * other.e01 + self.e01 * other.e11,
            self.e10 * other.e00 + self.e11 * other.e10,
            self.e10 * other.e01 + self.e11 * other.e11
        )

    def __imul__(self, other):
        result = self * other
        self.e00, self.e01, self.e10, self.e11 = result.e00, result.e01, result.e10, result.e11
        return self


def fibonacci_fastexp(n):
    step = M2x2(0, 1, 1, 1)
    fib = M2x2(0, 1, 1, 1)

    while n > 0:
        if n & 1:
            fib *= step
        step *= step
        n >>= 1

    return fib.e00

# FAST EXP BUT WITH KARATSUBA MULTIPLICATION
def karatsuba(x, y):
    if x < 10 or y < 10:
        return x * y

    n = max(len(str(x)), len(str(y)))
    m = n // 2

    high_x, low_x = divmod(x, 10 ** m)
    high_y, low_y = divmod(y, 10 ** m)

    z0 = karatsuba(low_x, low_y)
    z1 = karatsuba((low_x + high_x), (low_y + high_y))
    z2 = karatsuba(high_x, high_y)

    return z2 * 10 ** (2 * m) + (z1 - z2 - z0) * 10 ** m + z0


class M2x2Karatsuba:
    def __init__(self, e00, e01, e10, e11):
        self.e00 = e00
        self.e01 = e01
        self.e10 = e10
        self.e11 = e11

    def __mul__(self, other):
        return M2x2Karatsuba(
            karatsuba(self.e00, other.e00) + karatsuba(self.e01, other.e10),
            karatsuba(self.e00, other.e01) + karatsuba(self.e01, other.e11),
            karatsuba(self.e10, other.e00) + karatsuba(self.e11, other.e10),
            karatsuba(self.e10, other.e01) + karatsuba(self.e11, other.e11)
        )

    def __imul__(self, other):
        result = self * other
        self.e00, self.e01, self.e10, self.e11 = result.e00, result.e01, result.e10, result.e11
        return self


def fibonacci_karatsuba(n):
    step = M2x2Karatsuba(0, 1, 1, 1)
    fib = M2x2Karatsuba(0, 1, 1, 1)

    while n > 0:
        if n & 1:
            fib *= step
        step *= step
        n >>= 1

    return fib.e00

# Fast exp but FFT
def fft_multiply(x, y):
    x_digits = list(map(int, str(x)))[::-1]
    y_digits = list(map(int, str(y)))[::-1]

    n = 1
    while n < len(x_digits) + len(y_digits):
        n *= 2

    x_digits += [0] * (n - len(x_digits))
    y_digits += [0] * (n - len(y_digits))

    X = np.fft.fft(x_digits)
    Y = np.fft.fft(y_digits)
    Z = X * Y
    result_digits = np.fft.ifft(Z).real.round().astype(int)

    carry = 0
    for i in range(len(result_digits)):
        result_digits[i] += carry
        carry = result_digits[i] // 10
        result_digits[i] %= 10

    while carry:
        result_digits.append(carry % 10)
        carry //= 10

    return int("".join(map(str, result_digits[::-1])))


class M2x2FFT:
    def __init__(self, e00, e01, e10, e11):
        self.e00 = e00
        self.e01 = e01
        self.e10 = e10
        self.e11 = e11

    def __mul__(self, other):
        return M2x2FFT(
            fft_multiply(self.e00, other.e00) + fft_multiply(self.e01, other.e10),
            fft_multiply(self.e00, other.e01) + fft_multiply(self.e01, other.e11),
            fft_multiply(self.e10, other.e00) + fft_multiply(self.e11, other.e10),
            fft_multiply(self.e10, other.e01) + fft_multiply(self.e11, other.e11)
        )

    def __imul__(self, other):
        result = self * other
        self.e00, self.e01, self.e10, self.e11 = result.e00, result.e01, result.e10, result.e11
        return self


def fibonacci_fft(n):
    step = M2x2FFT(0, 1, 1, 1)
    fib = M2x2FFT(0, 1, 1, 1)

    while n > 0:
        if n & 1:
            fib *= step
        step *= step
        n >>= 1

    return fib.e00

# Fast double
def fibonacci_fast_double(n: int) -> int:
    def fib_pair(k: int) -> tuple[int, int]:
        """ returns F(k), F(k+1) """
        if k <= 1:
            return (k, 1)
        fh, fh1 = fib_pair(k>>1)
        fk = fh * ((fh1<<1) - fh)
        fk1 = fh1**2 + fh**2
        if k & 1:
            fk, fk1 = fk1, fk+fk1
        return fk, fk1

    return fib_pair(n)[0]

# Memoization
@lru_cache(maxsize=None)
def fibonacci_memo(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci_memo(n - 1) + fibonacci_memo(n - 2)

def measure_execution_times(func, test_indexes):
    results = []
    for index in test_indexes:
        start_time = time.time()
        func(index)
        elapsed_time = time.time() - start_time
        results.append(elapsed_time)

    print("Index:    ", "  ".join(map(str, test_indexes)))
    print("Time (ms):", "  ".join(f"{t*1000:.4f}" for t in results))
    return test_indexes, results


def plot_execution_times(test_indexes, execution_times):
    plt.plot(test_indexes, execution_times, marker='o', linestyle='-')
    plt.xlabel('n-th Fibonacci Term')
    plt.ylabel('Execution Time (ms)')
    plt.title('Fibonacci Memoization Time Analysis')
    plt.grid()
    plt.show()


# Example usage
print("Fast exp")
measure_execution_times(fibonacci_fastexp, compare_indexes)
print("Fast double")
measure_execution_times(fibonacci_fast_double, compare_indexes)
print("Memoization")
n_values, times = measure_execution_times(fibonacci_memo, compare_indexes)
plot_execution_times(n_values, times)

