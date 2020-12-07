#!/usr/bin/env python
import pytest
import os.path
from run import main

def find_json_files_in_dir(path):
    return [f.path for f in os.scandir(path) if f.is_file() and f.name.endswith('.json')]

positive_test_files = find_json_files_in_dir(os.path.join('.','regression-test','solvable'))
negative_test_files = find_json_files_in_dir(os.path.join('.','regression-test','unsolvable'))

@pytest.mark.parametrize('file', positive_test_files)
def test_positive(file):
    assert main(f'--input-json {file}'.split()) == 0

@pytest.mark.parametrize('file', negative_test_files)
def test_negative(file):
    assert main(f'--input-json {file}'.split(' ')) == 1
