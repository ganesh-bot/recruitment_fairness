# setup.py

from setuptools import find_packages, setup

setup(
    name="recruitment_fairness",
    version="0.1.0",
    description="Recruitment‐aware trial outcome prediction with fairness modules",
    author="Ganesh Pathak",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        # Because CI already does `pip install -r requirements.txt`,
        # you can leave this empty or duplicate core deps if you like.
    ],
)
