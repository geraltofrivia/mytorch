from setuptools import setup, find_packages

with open("README.md", "r") as readmefile:
    package_description = readmefile.read()

setup(
    name="my-torch",
    version="0.0.5",
    author="Priyansh Trivedi",
    author_email="mail@priyansh.page",
    description="A transparent boilerplate + bag of tricks to ease my (yours?) (our?) PyTorch dev time.",
    long_description=package_description,
    long_description_content_type="text/markdown",
    url="https://github.com/geraltofrivia/mytorch/",
    packages=find_packages('src'),
    package_dir={'': 'src'},
    project_urls={
        "Source Code": "https://github.com/geraltofrivia/mytorch"
    },
    install_requires=['spacy', 'tqdm', 'numpy'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix"
    ],
    keywords=[
        'deep learning', 'pytorch', 'boilerplate', 'machine learning', 'neural network', 'preprocessing'
    ]
)
