from timeit import default_timer as timer
import numpy as np
import copy

def quick_sort(array, low, high):
    if low < high:
        pi = partition(array, low, high)
        quick_sort(array, low, pi - 1)
        quick_sort(array, pi + 1, high)


def partition(array, low, high):
    pivot = array[high]
    i = low - 1
    for j in range(low, high):
        if array[j] <= pivot:
            i += 1
            array[i], array[j] = array[j], array[i]

    array[i + 1], array[high] = array[high], array[i + 1]
    return i + 1


def merge_sort(array, left, right):
    if left < right:
        middle = (left + right) // 2
        merge_sort(array, left, middle)
        merge_sort(array, middle + 1, right)
        merge(array, left, middle, right)


def merge(array, left, middle, right):
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

    while i < n1:
        array[k] = L[i]
        i += 1
        k += 1

    while j < n2:
        array[k] = R[j]
        j += 1
        k += 1


def heapify(arr, n, i):
    largest = i  # Initialize largest as root
    l = 2 * i + 1  # left = 2*i + 1
    r = 2 * i + 2  # right = 2*i + 2

    # See if left child of root exists and is
    # greater than root

    if l < n and arr[i] < arr[l]:
        largest = l

    # See if right child of root exists and is
    # greater than root

    if r < n and arr[largest] < arr[r]:
        largest = r

    # Change root, if needed

    if largest != i:
        (arr[i], arr[largest]) = (arr[largest], arr[i])  # swap

        # Heapify the root.

        heapify(arr, n, largest)


# The main function to sort an array of given size

def heap_sort(arr):
    n = len(arr)

    # Build a maxheap.
    # Since last parent will be at (n//2) we can start at that location.

    for i in range(n // 2, -1, -1):
        heapify(arr, n, i)

    # One by one extract elements

    for i in range(n - 1, 0, -1):
        (arr[i], arr[0]) = (arr[0], arr[i])  # swap
        heapify(arr, i, 0)


MIN_MERGE = 32


def calcMinRun(n):
    """Returns the minimum length of a
    run from 23 - 64 so that
    the len(array)/minrun is less than or
    equal to a power of 2.

    e.g. 1=>1, ..., 63=>63, 64=>32, 65=>33,
    ..., 127=>64, 128=>32, ...
    """
    r = 0
    while n >= MIN_MERGE:
        r |= n & 1
        n >>= 1
    return n + r


# This function sorts array from left index to
# to right index which is of size atmost RUN
def insertionSort(arr, left, right):
    for i in range(left + 1, right + 1):
        j = i
        while j > left and arr[j] < arr[j - 1]:
            arr[j], arr[j - 1] = arr[j - 1], arr[j]
            j -= 1


# Merge function merges the sorted runs
def merge_tim(arr, l, m, r):
    # original array is broken in two parts
    # left and right array
    len1, len2 = m - l + 1, r - m
    left, right = [], []
    for i in range(0, len1):
        left.append(arr[l + i])
    for i in range(0, len2):
        right.append(arr[m + 1 + i])

    i, j, k = 0, 0, l

    # after comparing, we merge those two array
    # in larger sub array
    while i < len1 and j < len2:
        if left[i] <= right[j]:
            arr[k] = left[i]
            i += 1

        else:
            arr[k] = right[j]
            j += 1

        k += 1

    # Copy remaining elements of left, if any
    while i < len1:
        arr[k] = left[i]
        k += 1
        i += 1

    # Copy remaining element of right, if any
    while j < len2:
        arr[k] = right[j]
        k += 1
        j += 1


# Iterative Timsort function to sort the
# array[0...n-1] (similar to merge sort)
def tim_sort(arr):
    n = len(arr)
    minRun = calcMinRun(n)

    # Sort individual subarrays of size RUN
    for start in range(0, n, minRun):
        end = min(start + minRun - 1, n - 1)
        insertionSort(arr, start, end)

        # Start merging from size RUN (or 32). It will merge
    # to form size 64, then 128, 256 and so on ....
    size = minRun
    while size < n:

        # Pick starting point of left sub array. We
        # are going to merge arr[left..left+size-1]
        # and arr[left+size, left+2*size-1]
        # After every merge, we increase left by 2*size
        for left in range(0, n, 2 * size):

            # Find ending point of left sub array
            # mid+1 is starting point of right sub array
            mid = min(n - 1, left + size - 1)
            right = min((left + 2 * size - 1), (n - 1))

            # Merge sub array arr[left.....mid] &
            # arr[mid+1....right]
            if mid < right:
                merge_tim(arr, left, mid, right)

        size = 2 * size


    # arr = [1, 5, 2, 5, 6, 8, 4, 0, -1]
# quick_sort(arr, 0, len(arr) - 1)
# merge_sort(arr, 0, len(arr) - 1)
SIZE = 997
SORTED_RATIO = 0.2
TEST_RUNS = 10

labels = (
    "completely random",
    f"same values x{SIZE}",
    "sorted in reverse",
    "correctly sorted",
    "partially sorted in the beginning",
    "partially sorted in the end",
    "partially sorted in the middle"
)

test_cases = (
    # completely random
    lambda _: np.random.randint(-1000, 1000, SIZE),

    # same values x10000
    lambda _: np.full(SIZE, 1),

    # sorted in reverse
    lambda _: np.arange(SIZE, 0, -1),

    # correctly sorted
    lambda _: np.arange(1, SIZE + 1),

    # partially sorted in the beginning
    lambda _: np.concatenate((np.arange(1, SIZE * SORTED_RATIO + 1),
     np.random.randint(-1000, 1000, int((1 - SORTED_RATIO) * SIZE))
    )),

    # partially sorted in the end
    lambda _: np.concatenate((np.random.randint(-1000, 1000, int((1 - SORTED_RATIO) * SIZE)),
     np.arange(1, SIZE * SORTED_RATIO + 1)
    )),

    # partially sorted in the middle
    lambda _: np.concatenate((np.random.randint(-1000, 1000, int((1 - SORTED_RATIO / 2) * SIZE))
     , np.arange(1, SIZE * SORTED_RATIO + 1)
     , np.random.randint(-1000, 1000, int((1 - SORTED_RATIO / 2) * SIZE))
     )),
)


def sort_wrapper(func):
    return lambda arr: func(arr, 0, SIZE - 1)


sorting_algs = (
    sort_wrapper(quick_sort),
    sort_wrapper(merge_sort),
    heap_sort,
    tim_sort
)

sorting_labels = (
    "Quick Sort",
    "Merge Sort",
    "Heap Sort",
    "Tim Sort"
)

num_sorting_algs = len(sorting_algs)
num_test_cases = len(test_cases)
exec_mat = np.zeros((num_sorting_algs, num_test_cases))

for k, sort in enumerate(sorting_algs):
    print()
    print(sorting_labels[k])
    for i, test_arr in enumerate(test_cases):
        exec_times = []
        for j in range(TEST_RUNS):
            arr = copy.copy(test_arr(None))
            # print(arr[:5])
            start = timer()
            # heap_sort(test_arr)
            # quick_sort(test_arr, 0, SIZE - 1)
            # merge_sort(test_arr, 0, SIZE - 1)
            sort(arr)
            end = timer()
            exec_times.append(end - start)

        exec_mat[k, i] = np.average(exec_times) * 1000
        print(f"{labels[i]}: {exec_mat[k, i]}ms")

    print(f"Average speed: {np.average(exec_mat[k])}ms")

print()
print("Top Performers: ")
min_indices = np.argmin(exec_mat, axis=0)
for i, label in enumerate(labels):
    print(f"{label}: {sorting_labels[min_indices[i]]} ({exec_mat[min_indices[i]][i]})")

print()
print("Worst Performers: ")
min_indices = np.argmax(exec_mat, axis=0)
for i, label in enumerate(labels):
    print(f"{label}: {sorting_labels[min_indices[i]]} ({exec_mat[min_indices[i]][i]})")

def universal_sort(arr):
    is_sorted = all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1))
    if is_sorted:
        return arr

    is_reverse_sorted = all(arr[i] >= arr[i + 1] for i in range(len(arr) - 1))
    if is_reverse_sorted:
        # print("Sorted in reverse, use Merge Sort")
        # Merge Sort should be used here
        merge_sort(arr, 0, len(arr)-1)
        return arr

    first_element = arr[0]
    if all(x == first_element for x in arr):
        # print("All elements are the same, use Heap Sort")
        # Heap Sort should be used here
        # return arr  # No need to sort, all elements are identical
        heap_sort(arr)
        return arr

    increasing_runs = []
    decreasing_runs = []
    start = 0
    i = 1

    while i < len(arr):
        if arr[i] > arr[i - 1]:  # increasing
            while i < len(arr) and arr[i] > arr[i - 1]:
                i += 1
            increasing_runs.append((start, i - 1))
            start = i
        elif arr[i] < arr[i - 1]:  # decreasing
            while i < len(arr) and arr[i] < arr[i - 1]:
                i += 1
            decreasing_runs.append((start, i - 1))
            start = i
        else:  # equal
            i += 1
            start = i


    all_runs = increasing_runs + decreasing_runs
    if all_runs:
        longest_run = max(all_runs, key=lambda x: x[1] - x[0])
        run_length = longest_run[1] - longest_run[0] + 1

        if run_length == len(arr):
            return arr if is_sorted else merge_sort(arr, 0, len(arr)-1)
        elif longest_run[0] == 0:
            # print("Partially sorted at the beginning, use Quick Sort")
            # Quick Sort should be used here
            quick_sort(arr, 0, len(arr) - 1)
            return arr
        elif longest_run[0] in range(len(arr) - int(len(arr) * 0.1), len(arr) - 1):
            # print("Partially sorted at the end, use Merge Sort")
            # Merge Sort should be used here
            merge_sort(arr, 0, len(arr) - 1)
            return arr
        else:
            # print("Partially sorted in the middle, use Merge Sort")
            # Merge Sort should be used here
            merge_sort(arr, 0, len(arr) - 1)
            return arr

    merge_sort(arr, 0, len(arr) - 1)
    return arr


print()
print("Universal Sort")
for i, test_arr in enumerate(test_cases):
    exec_times = []
    for j in range(TEST_RUNS):
        arr = copy.copy(test_arr(None))
        start = timer()
        universal_sort(arr)
        end = timer()
        exec_times.append(end - start)

    exec_mat[k, i] = np.average(exec_times) * 1000
    print(f"{labels[i]}: {exec_mat[k, i]}ms")
