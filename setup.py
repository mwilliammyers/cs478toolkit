from setuptools import setup

setup(
    name='lathe',
    version='0.2.1',
    description='Basic machine learning toolkit for BYU CS478',
    url='http://github.com/mwilliammyers/lathe',
    author='mwilliammyers',
    author_email='mwilliammyers@gmail.com',
    license='MIT',
    packages=['lathe'],
    install_requires=['liac-arff', 'numpy', 'scikit-learn', 'scipy'],
    keywords=['machine learning', 'ml', 'toolkit', 'byu', 'cs478'],
    zip_safe=False)
