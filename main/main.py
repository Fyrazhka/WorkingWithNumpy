import numpy as np
import matplotlib.pyplot as plt


def task1():
    """
        Create two arrays a and b using np.random.randint.
        Perform the following operations and output the results:
        Array addition
        Subtracting Arrays
        Element-wise array multiplication
        Element-wise division of arrays
        Find the maximum and minimum value in each array.
        Find the mean and standard deviation for each array.
    """
    a = np.random.randint(0, 100, 5)
    b = np.random.randint(0, 100, 5)
    sum_ab = a+b
    diff_ab = a-b
    prod_ab = a*b
    quot_ab = a/b
    print("Array a:", a)
    print("Array b:", b)
    print("Sum:", sum_ab)
    print("Difference:", diff_ab)
    print("Product:", prod_ab)
    print("Quotient:", quot_ab)
    print(f'Min(a) {np.min(a)} Max(a) {np.max(a)}\nMin(b) {np.min(b)} Max(b) {np.max(b)}')
    print(f'Mean(a) {np.mean(a)} Std(a) {np.std(a)}\nMean(b) {np.mean(b)} Std(b) {np.std(b)}')


def task2():
    """
        Create a matrix using np.random.randint.
        Find the sum of all elements in the matrix.
        Find the average for each row and each column.
        Transpose the matrix.
        Find the determinant of the matrix and its inverse (if it exists).
    """
    matrix = np.random.randint(0, 100, (4, 4))
    sum_all = np.sum(matrix)
    mean_rows = np.mean(matrix, axis=1)
    mean_cols = np.mean(matrix, axis=0)
    transpose_matrix = np.transpose(matrix)

    det_matrix = np.linalg.det(matrix)
    inverse_matrix = np.linalg.inv(matrix) if det_matrix != 0 else None

    print("Matrix:\n", matrix)
    print("Sum of all elements:", sum_all)
    print("Mean of each row:", mean_rows)
    print("Mean of each column:", mean_cols)
    print("Transposed matrix:\n", transpose_matrix)
    print("Determinant of the matrix:", det_matrix)
    if inverse_matrix is not None:
        print("Inverse matrix:\n", inverse_matrix)
    else:
        print("Matrix is singular and does not have an inverse.")


def task3():
    """
    Create an array using np.random.randint.
    Extract elements from 2nd and 4th columns.
    Extract elements from 3rd and 5th rows.
    Extract a 3x3 subarray from the center of the array.
    Change the values of the elements in the 2x2 subarray located in the upper left corner of the array to 0.
    """
    array = np.random.randint(0, 100, (6, 6))
    array_columns_elements = array[:, [1, 3]]
    array_rows_elements = array[[2, 4], :]
    array_center_elements = array[1: 4, 1:4]
    print("Original array:\n", array)
    print("Columns 2 and 4:\n", array_columns_elements)
    print("Rows 3 and 5:\n", array_rows_elements)
    print("Center 3x3 subarray:\n", array_center_elements)

    array[0:2, 0:2] = 0
    print("Modified array with top-left 2x2 subarray set to 0:\n", array)


def task4():
    """
    Create an array x containing 100 evenly distributed values from 0 to 2π using np.linspace.
    Calculate the sine and cosine values for all elements of the array x.
    Plot graphs for sine and cosine on the same graph.
    Add a title, axis labels, and legend to differentiate your graphs.
    """
    x = np.linspace(0, 2*np.pi, 100)
    x_sin = np.sin(x)
    x_cos = np.cos(x)
    plt.plot(x, x_sin, label='sin(x)')
    plt.plot(x, x_cos, label='cos(x)')
    plt.title('Sin and Cosine on the same graph')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    task1()
    task2()
    task3()
    task4()
