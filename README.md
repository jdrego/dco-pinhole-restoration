# Deep Camera Obscura: An Image Restoration Pipeline for Pinhole Photography

Repository for our paper: 
  
_"Deep Camera Obscura: An Image Restoration Pipeline for Pinhole Photography"_, Joshua D. Rego<sup>1</sup>, Huaijin Chen<sup>2</sup>, Shuai Li<sup>2</sup>, Jinwei Gu<sup>2</sup>, Suren Jayasuriya<sup>1</sup>

<sup><sup>1</sup> Arizona State University | <sup>2</sup> Sensebrain Technology</sup>

### Dataset
* Simulated pinhole HDR+ dataset can be downloaded here: [Google Drive](https://drive.google.com/drive/folders/1Pakes8BFpNjmzUK494MYUZ0WMqI-aRSx?usp=sharing)


## Requirements:

The python package requirements are included in the file `requirements.yml`. We recommend installing the required packages in a separate virtual python environment using Anaconda: 

[Conda Download](https://www.anaconda.com/products/individual), 
[Conda Installation Instruction](https://docs.anaconda.com/anaconda/install/index.html)

After conda is correctly installed, run `conda activate` in the terminal, then make sure you are inside the project folder in the terminal and run the following command line which will create a new virtual environment named `DCO`: 

    conda env create -f requirements.yml

After all packages are installed you can switch into this virtual environment with:
    
    conda activate DCO

## Instructions:

To run example the DCO pipeline on the default example images, run the following in a command line at the root project folder:

    python run.py

Optionally a Python Notebook file, `DCO_pipeline.ipynb`, is also included to easier visualize the images through the DCO pipeline. 