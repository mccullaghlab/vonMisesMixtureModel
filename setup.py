
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='vonMisesMixtureModel',
    version='0.0.0',
    author='Martin McCullagh',
    author_email='martin.mccullagh@okstate.edu',
    description='python/pyTorch code for determining parameters for bivariate von Mises mixture models',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/mccullaghlab/vonMisesMixtureModel',
    project_urls = {
        "Bug Tracker": "https://github.com/mccullaghlab/vonMisesMixtureModel/issues"
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    license='MIT',
    install_requires=['numpy','torch', 'scipy'],
)
