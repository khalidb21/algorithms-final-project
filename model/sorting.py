# Merge sort 
def mergeSort(array):
    if len(array) > 1: 
        left = []
        right = []
        mid = len(array) // 2

        for x in range(0, mid):
            left.append(array[x])
        for x in range(mid + 1, len(array)):
            right.append(array[x])
       
        mergeSort(left)
        mergeSort(right)

        i = 0
        j = 0
        k = 0
 
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                array[k] = left[i]                
                i += 1
            else:
                array[k] = right[j]
                j += 1
            k += 1
     
        while i < len(left):
            array[k] = left[i]
            i += 1
            k += 1
 
        while j < len(right):
            array[k] = right[j]
            j += 1
            k += 1

def mergeSort2D(array):
    array_copy = array.copy() #create a copy so original data is not changed

    for x in range(len(array_copy)): #loop through all batteries
        mergeSort(array_copy[x])
    return array_copy

# Selection Sort
def selectionSort(row):
    n = len(row)
    for i in range(n - 1):
        min = i
        for j in range(i + 1, n):
            if row[j] < row[min]:
                min = j
        if min != i:
            row[i], row[min] = row[min], row[i]

def selectionSort2D(array_2d):
    b = array_2d.copy()         
    for r in range(len(b)):     
        selectionSort(b[r])
    return b


# def compare_sorting_methods(df):
