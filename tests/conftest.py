def pytest_configure(config):
    config.addinivalue_line("markers", "slow: loads a real model or needs the network (deselect with -m 'not slow')")
