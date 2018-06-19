from setuptools import setup
from setuptools import find_packages


version = '0.1'

setup(name='keras-text',
      version=version,
      description='Text Classification Library for Keras',
      author='Raghavendra Kotikalapudi, Johannes Filter',
      author_email='ragha@outlook.com, hi@jfilter.de',
      url='https://github.com/jfilter/text-classification-keras',
      download_url='https://github.com/jfilter/text-classification-keras/tarball/{}'.format(
          version),
      license='MIT',
      install_requires=['keras>=2.1.2', 'six',
                        'spacy>=2.0.3', 'scikit-learn', 'joblib', 'jsonpickle', 'pickle'],
      extras_require={
          'tests': ['pytest',
                    'pytest-pep8',
                    'pytest-xdist',
                    'pytest-cov'],
      },
      include_package_data=True,
      packages=find_packages())
