# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import numpy as np
from typing import List
from typing import Dict
from typing import Any


def bulk_transform(partition: np.ndarray, field_name: str, action,
                   offset: int) -> List[Dict[str, Any]]:
    """Partitions and transforms a list of vectors into OpenSearch's bulk
    injection format.
    Args:
        offset: to start counting from
        partition: An array of vectors to transform.
        field_name: field name for action
        action: Bulk API action.
    Returns:
        An array of transformed vectors in bulk format.
    """
    actions = []
    _ = [
        actions.extend([action(i + offset), None])
        for i in range(len(partition))
    ]
    actions[1::2] = [{field_name: vec} for vec in partition.tolist()]
    return actions


def parse_string_parameter(key: str, params: dict, default: str = None) -> str:
    if key not in params:
        if default is not None:
            return default
        raise ConfigurationError(
            "Value cannot be None for param {}".format(key)
        )

    if type(params[key]) is str:
        return params[key]

    raise ConfigurationError("Value must be a string for param {}".format(key))


def parse_int_parameter(key: str, params: dict, default: int = None) -> int:
    if key not in params:
        if default:
            return default
        raise ConfigurationError(
            "Value cannot be None for param {}".format(key)
        )

    if type(params[key]) is int:
        return params[key]

    raise ConfigurationError("Value must be a int for param {}".format(key))

def parse_float_parameter(key: str, params: dict, default: float = None) -> float:
    if key not in params:
        if default:
            return default
        raise ConfigurationError(
            "Value cannot be None for param {}".format(key)
        )
 
    if type(params[key]) is float:
        return params[key]
 
    raise ConfigurationError("Value must be a float for param {}".format(key))


class ConfigurationError(Exception):
    """Exception raised for errors configuration.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message: str):
        self.message = f'{message}'
        super().__init__(self.message)
