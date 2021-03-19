import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="backprop",
    version="0.0.2",
    author="Backprop",
    author_email="hello@backprop.co",
    description="Backprop",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/backprop-ai/backprop",
    packages=setuptools.find_packages(),
    install_requires=[
        "transformers>=4.3.2",
        "sentence_transformers>=0.4.1.2",
        "torch<1.8.0",
        "torchvision",
        "ftfy",
        "pytorch_lightning>=1.2.0",
        "dill"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
