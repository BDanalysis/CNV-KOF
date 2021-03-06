# CNV-KOF
CNV-KOF: a new approach to detect copy number variations from next-generation sequencing data

## Installation
The following softwares must be installed on your machine:
Python : tested with version 3.7
R: tested with version 4.0.5

### Python dependencies
* numpy 
* numba
* sklearn
* pandas
* pysam
* rpy2

You can install the above package using the following command：
pip install numpy numba sklearn pandas pysam rpy2


### R packages
* DNAcopy

To install this package, start R and enter:
>install.packages("BiocManager")
>BiocManager::install("DNAcopy")


## Running
CNV-KOF requires two input files, a bam file and a reference folder,
the reference folder contains the reference sequence for each chromosome of the bam file.

### runnig command
python CNV-KOF.py [bamfile] [reference] [output] [binSize] [segCount] [k] [h]

·bamfile: a bam file
·reference: the reference folder path
·output:the the output file path
·binSize: the window size ('1000'by default)
·segCount: the number of  partitions to CBS ('50' by default)
·k: the k-distance parameter of the KOF ('10' by default)
·h: the bandwidth parameter of the KOF ('0.2' by default)

### run the default example
python CNV-KOF.py test.bam reference result 1000 50 10 0.2

Finally, two result files will be produced, one is the mutation information of detection and the other is the statistical data.
