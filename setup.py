from setuptools import setup, find_packages

setup(
    name="cell-aging-rejuvenation",
    version="1.0.0",
    description="Deep learning pipeline for simulating MSC cellular senescence and rejuvenation using CycleGAN and Latent Diffusion Models",
    author="Emir Muhammet Aran",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'torchaudio>=2.0.0',
        'diffusers>=0.21.0',
        'transformers>=4.30.0',
        'Pillow>=9.0.0',
        'numpy>=1.21.0',
        'pyyaml>=6.0',
        'tqdm>=4.65.0',
        'opencv-python>=4.7.0',
        'scikit-learn>=1.3.0',
        'pytorch-fid>=0.3.0',
        'accelerate>=0.20.0',
        'peft>=0.4.0',
    ],
    entry_points={
        'console_scripts': [
            'train-cyclegan=src.training.cyclegan.train_cyclegan:main',
            'train-classifier=src.training.classifier.train_classifier:main',
            'train-ldm=src.training.ldm.train_ldm:main',
            'infer-cyclegan=src.inference.cyclegan.generate_all:main',
            'eval-classifier=src.inference.classifier.evaluate_classifier:main',
        ],
    },
)
