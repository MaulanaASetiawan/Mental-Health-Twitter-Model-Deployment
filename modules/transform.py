"""
Transform module for preprocessing the input features.
"""
import tensorflow as tf

LABEL_KEY = "label"
FEATURE_KEY = "post_text"


def transformed_name(key):
    """
    Generate the name of the transformed feature.

    Args:
        key: The name of the feature.

    Returns:
        A string representing the transformed feature name.
    """
    return key + "_xf"


def preprocessing_fn(inputs):
    """
    Preprocess the input features.

    Args:
        inputs: Dictionary containing the input features.

    Returns:
        Dictionary containing the transformed features.
    """
    outputs = {}
    outputs[transformed_name(FEATURE_KEY)] = tf.strings.lower(
        inputs[FEATURE_KEY])
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)
    return outputs
