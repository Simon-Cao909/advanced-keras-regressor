def shapes_equal(a, b):
    '''
    Checks if the shapes of two arrays are equal

    :param a (array-like): The first array
    :param b (array-like): The second array

    :returns (bool): True if the shapes are equal
                     False otherwise
    '''
    if len(a) != len(b):
        return False
    return all(
        (x == y) or (x is None) or (y is None)
        for x, y in zip(a, b)
    )