from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))
readme_path = os.path.join(here, "README.md")
with codecs.open(readme_path, encoding="utf-8") as fh:
    long_description = fh.read()
VERSION = '0.0.1'
DESCRIPTION = 'Tabular-Infused Parameter Efficient Finetuning (tipeft)'
LONG_DESCRIPTION = "Tabular-Infused Parameter Efficient Finetuning (tipeft) specifically designed for postoperative risk prediction using clinical notes and complementary preoperative tabular features. Available for re-parameterization methods (LoRa and IA3)."

# Setting up
setup(
    name="tipeft",
    version=VERSION,
    author="Charles Alba",
    author_email="alba@wustl.edu",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=[],
    keywords=['Parameter Efficient Finetuning',"AI in Medicine","AI in Healthcare","Postoperative Risk Prediction"],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)