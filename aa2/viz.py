from timeit import default_timer as timer
import numpy as np
import copy
import matplotlib.pyplot as plt

# Sorting Algorithms
def quick_sort(array, low, high, ax=None):
    if low < high:
        pi = partition(array, low, high, ax)
        quick_sort(array, low, pi - 1, ax)
        quick_sort(array, pi + 1, high, ax)


def partition(array, low, high, ax):
    pivot = array[high]
    i = low - 1
    for j in range(low, high):
        if array[j] <= pivot:
            i += 1
            array[i], array[j] = array[j], array[i]
            if ax:
                visualize(array, ax, i, j)

    array[i + 1], array[high] = array[high], array[i + 1]
    if ax:
        visualize(array, ax, i + 1, high)
    return i + 1


def merge_sort(array, left, right, ax=None):
    if left < right:
        middle = (left + right) // 2
        merge_sort(array, left, middle, ax)
        merge_sort(array, middle + 1, right, ax)
        merge(array, left, middle, right, ax)


def merge(array, left, middle, right, ax):
    n1 = middle - left + 1
    n2 = right - middle
    L = [0] * n1
    R = [0] * n2

    for i in range(0, n1):
        L[i] = array[left + i]

    for j in range(0, n2):
        R[j] = array[middle + 1 + j]

    i, j, k = 0, 0, left
    while i < n1 and j < n2:
        if L[i] <= R[j]:
            array[k] = L[i]
            i += 1
        else:
            array[k] = R[j]
            j += 1

        k += 1
        if ax:
            visualize(array, ax, k)

    while i < n1:
        array[k] = L[i]
        i += 1
        k += 1
        if ax:
            visualize(array, ax, k)

    while j < n2:
        array[k] = R[j]
        j += 1
        k += 1
        if ax:
            visualize(array, ax, k)


def heapify(arr, n, i, ax=None):
    largest = i
    l = 2 * i + 1
    r = 2 * i + 2

    if l < n and arr[i] < arr[l]:
        largest = l

    if r < n and arr[largest] < arr[r]:
        largest = r

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        if ax:
            visualize(arr, ax, i, largest)
        heapify(arr, n, largest, ax)


def heap_sort(arr, ax=None):
    n = len(arr)

    for i in range(n // 2, -1, -1):
        heapify(arr, n, i, ax)

    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        if ax:
            visualize(arr, ax, i, 0)
        heapify(arr, i, 0, ax)


def tim_sort(arr, ax=None):
    n = len(arr)
    minRun = calcMinRun(n)

    for start in range(0, n, minRun):
        end = min(start + minRun - 1, n - 1)
        insertionSort(arr, start, end, ax)

    size = minRun
    while size < n:
        for left in range(0, n, 2 * size):
            mid = min(n - 1, left + size - 1)
            right = min((left + 2 * size - 1), (n - 1))
            if mid < right:
                merge_tim(arr, left, mid, right, ax)
        size = 2 * size


def insertionSort(arr, left, right, ax=None):
    for i in range(left + 1, right + 1):
        j = i
        while j > left and arr[j] < arr[j - 1]:
            arr[j], arr[j - 1] = arr[j - 1], arr[j]
            j -= 1
            if ax:
                visualize(arr, ax, j)


def merge_tim(arr, l, m, r, ax=None):
    len1, len2 = m - l + 1, r - m
    left, right = [], []
    for i in range(0, len1):
        left.append(arr[l + i])
    for i in range(0, len2):
        right.append(arr[m + 1 + i])

    i, j, k = 0, 0, l
    while i < len1 and j < len2:
        if left[i] <= right[j]:
            arr[k] = left[i]
            i += 1
        else:
            arr[k] = right[j]
            j += 1
        k += 1
        if ax:
            visualize(arr, ax, k)

    while i < len1:
        arr[k] = left[i]
        k += 1
        i += 1
        if ax:
            visualize(arr, ax, k)

    while j < len2:
        arr[k] = right[j]
        k += 1
        j += 1
        if ax:
            visualize(arr, ax, k)


MIN_MERGE = 32
def calcMinRun(n):
    r = 0
    while n >= MIN_MERGE:
        r |= n & 1
        n >>= 1
    return n + r


# Visualization Function
def visualize(array, ax, *highlight):
    ax.clear()
    ax.bar(range(len(array)), array, color="blue")
    for h in highlight:
        ax.bar(h, array[h], color="red")
    plt.pause(0.01)


# Wrapper Function for Visualization
def visualize_sorting(sort_func, arr):
    fig, ax = plt.subplots()
    ax.set_title(f"{sort_func.__name__} Visualization")
    sort_func(arr, ax=ax)
    plt.show()


# Test Cases and Execution
SIZE = 50  # Smaller size for better visualization
SORTED_RATIO = 0.2
TEST_RUNS = 1  # Only one run for visualization
MIN_VALUE = 0
MAX_VALUE = 10_000

test_cases = (
    lambda _: np.random.randint(MIN_VALUE, MAX_VALUE, SIZE),
    lambda _: np.full(SIZE, 1),
    lambda _: np.arange(SIZE, 0, -1),
    lambda _: np.arange(1, SIZE + 1),
    lambda _: np.concatenate((np.arange(1, SIZE * SORTED_RATIO + 1),
                             np.random.randint(MIN_VALUE, MAX_VALUE, int((1 - SORTED_RATIO) * SIZE)))),
    lambda _: np.concatenate((np.random.randint(MIN_VALUE, MAX_VALUE, int((1 - SORTED_RATIO) * SIZE)),
                              np.arange(1, SIZE * SORTED_RATIO + 1))),
    lambda _: np.concatenate((np.random.randint(MIN_VALUE, MAX_VALUE, int((1 - SORTED_RATIO / 2) * SIZE)),
                              np.arange(1, SIZE * SORTED_RATIO + 1),
                              np.random.randint(MIN_VALUE, MAX_VALUE, int((1 - SORTED_RATIO / 2) * SIZE)))),
)

sorting_algs = (
    lambda arr, ax=None: quick_sort(arr, 0, len(arr) - 1, ax),
    lambda arr, ax=None: merge_sort(arr, 0, len(arr) - 1, ax),
    lambda arr, ax=None: heap_sort(arr, ax),
    lambda arr, ax=None: tim_sort(arr, ax),
)

# Visualize each sorting algorithm
for sort in sorting_algs:
    arr = test_cases[0](None)  # Use the first test case for visualization
    visualize_sorting(sort, arr)