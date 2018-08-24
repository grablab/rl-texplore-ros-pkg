import tensorflow as tf
import numpy as np

def adjust_shape(placeholder, data):
    '''
    adjust shape of the data to the shape of the placeholder if possible.
    If shape is incompatible, AssertionError is thrown
    Parameters:
        placeholder     tensorflow input placeholder

        data            input data to be (potentially) reshaped to be fed into placeholder

    Returns:
        reshaped data
    '''

    if not isinstance(data, np.ndarray) and not isinstance(data, list):
        return data
    if isinstance(data, list):
        data = np.array(data)

    placeholder_shape = [x or -1 for x in placeholder.shape.as_list()]

    assert _check_shape(placeholder_shape, data.shape), \
        'Shape of data {} is not compatible with shape of the placeholder {}'.format(data.shape, placeholder_shape)

    return np.reshape(data, placeholder_shape)


def _check_shape(placeholder_shape, data_shape):
    ''' check if two shapes are compatible (i.e. differ only by dimensions of size 1, or by the batch dimension)'''
    squeezed_placeholder_shape = _squeeze_shape(placeholder_shape)
    squeezed_data_shape = _squeeze_shape(data_shape)

    for i, s_data in enumerate(squeezed_data_shape):
        s_placeholder = squeezed_placeholder_shape[i]
        if s_placeholder != -1 and s_data != s_placeholder:
            return False
    return True

def _squeeze_shape(shape):
    return [x for x in shape if x != 1]

