import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="eeg_emotion_recognition", # Replace with your own username
    version="0.0.1",
    author="Pratikshya Mishra",
    author_email="pratikshya.mishra72@gmail.com",
    description="To predict the emotion from EEG signals in the quadrant of valence, arousal and dominance.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject", #add github link after project uploaded there
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)