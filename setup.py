from setuptools import find_packages, setup

def get_requirements(path):
    requirements = []
    with open(path) as file:
        requirements = file.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        
        if "-e ." in requirements:
            requirements.remove("-e .")
            
    return requirements

setup(
    name = 'mlproject',
    version = '0.0.1',
    author = 'nourshaaban1',
    author_email = ["nourshaaban59@gmail.conm"],
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)
