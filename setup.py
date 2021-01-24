import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="kiri",
    version="0.4.0",
    author="Kiri OÜ",
    author_email="hello@kiri.ai",
    description="Kiri Natural Language Engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kiri-ai/kiri",
    packages=setuptools.find_packages(),
    install_requires=[
        "transformers>=4.0.0",
        "sentence_transformers",
        "nltk",
        "scipy",
        "shortuuid",
        "elasticsearch",
        "torch"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
