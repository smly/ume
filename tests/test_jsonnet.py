# -*- coding: utf-8 -*-
import json
import tempfile

import ume.externals.jsonnet
from ume.externals._jsonnet_wrapper import load, loads


class TestJsonnet(object):
    def test_jsonnet_loads(self):
        raw_string_json = """
/* This is a test. */
{
    "name": "value",  // test case
}
"""
        assert ume.externals.jsonnet.loads(raw_string_json) == {"name": "value"}

    def test_jsonnet_loads(self):
        with tempfile.NamedTemporaryFile('w') as f:
            f.write("""
{
    "name": "value",  // test case
}
""")
            f.flush()
            json_obj = ume.externals.jsonnet.load(f.name)
            assert json_obj == {"name": "value"}


class TestJsonnetCythonWrapper(object):
    def test_jsonnet_load(self):
        raw_string_json = loads("""
/* This is a test. */
{
    "name": "value",  // test case
}
""")
        json_obj = json.loads(raw_string_json.decode('utf-8'))

        assert json_obj == {"name": "value"}
