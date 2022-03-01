from setuptools import setup, find_packages
name = "KC Torch Experiment"
release = "0"
version = "0"
extensions = []

setup(
    name=name,
    version=version,
    description="Learning torch",
    url="",
    author="Keith Chow",
    license="MIT",
    packages=find_packages(),
    tests_require=["pytest"],
    setup_requires=[],
    install_requires=[
        "torch",
        "tensorboard",
        "numpy",
        "tqdm",             # progress bar
    ],
    dependency_links=[
    ],
    command_options={
        'build_sphinx': {
            'project': ('setup.py', name),
            'version': ('setup.py', version),
            'release': ('setup.py', release),
            'source_dir': ('setup.py', 'docs/source')
            }
        },
    )