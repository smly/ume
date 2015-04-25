# -*- coding: utf-8 -*-
"""
jsonnet <http://google.github.io/jsonnet/doc/> is a domanin specific
configuration language that helps you defin eJSON data.

Converting from Jsonnet into JSON::

    >>> from ume.externals import jsonnet
    >>> jsonnet.loads('''{"foo": "var",}''')
    {'foo': 'bar'}

"""
import json

from ume.externals import _jsonnet_wrapper


def loads(s):
    """
    Convert ``s`` (a ``s`` instance containing a Jsonnet document) to a JSON
    data.
    """
    raw_json_str = _jsonnet_wrapper.loads(s)
    json_obj = json.loads(raw_json_str.decode('utf-8'))

    return json_obj


def load(filename):
    raw_json_str = _jsonnet_wrapper.load(filename)
    json_obj = json.loads(raw_json_str.decode('utf-8'))

    return json_obj
