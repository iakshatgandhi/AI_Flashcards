from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()
    # Remove comments and empty lines
    requirements = [r for r in requirements if r and not r.startswith('#')]

setup(
    name="ai-flashcard-generator",
    version="0.1.0",
    author="AI Flashcard Team",
    author_email="example@example.com",
    description="AI-powered educational flashcard generator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ai-flashcard-generator",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
)
