# Graph Evolving Meta-Learning for Low-resource Medical Dialogue Generation

*Code and data coming soon...*

This repo contains the code of the following paper:

[Graph Evolving Meta-Learning for Low-resource Medical Dialogue Generation](TODO)

*Shuai Lin, Pan Zhou, Xiaodan Liang, Jianheng Tang, Ruihui Zhao, Ziliang Chen, Liang Lin.*    
*AAAI 2021*  

## Prerequisites

1. Allennlp (0.9.1-unreleased) 

2. pytorch == 1.4.0

3. Others should be found in ```./allennlp/requirements.txt```

```[Note]```: You need to install allennlp with the ```editable``` mode, i.e.,
```
cd ./allennlp
pip install --editable .
cd ..
```
since we have modified this toolkit (including added the ```metatrainer.py``` 
in the directory ```./allennlp/training``` and so on).

## Datasets

Please download both datasets from the google drive as follows:
```
wget https://drive.google.com/file/d/1KZ0CrIVZhSLxlZ-V5pnksvgH1xlyd54F/view?usp=sharing
tar zxvf cy.tar.gz
wget https://drive.google.com/file/d/1sZzb3Nzm_Z37lNCfgusJscFuiyhUON5j/view?usp=sharing
tar zxvf fd.tar.gz
```

1. ```CMDD```: The directory ```fd/dis_pk_dir```, which includes ```meta_train```
 and ```meta_test```. (The number of the file name represents the idx of a
  disease.)
  
2. ```MDG-Chunyu```: The directory ```cy/dis_pk_dir```, which also includes
 ```meta_train``` and ```meta_test```. 
 
## Quick Start

Most of the running commands are written in the script `run.sh`, which
follows the offical `train/fine-tune/evaluate` way of the allennlp. Take the
following one as an example:

[1]. Training:
```
CUDA_VISIBLE_DEVICES=1 allennlp train -s $save_directory$ \
  $config_file(.json)$ \
  --include-package $model_file$
```

[2]. Fine-tuning:
```
CUDA_VISIBLE_DEVICES=1 allennlp fine-tune -m $old save_directory$ \
  -c $config_file(.json)$ \
  --include-package $model_file$
  -s $new save_directory$
```  

[3]. Testing:
```
CUDA_VISIBLE_DEVICES=3 allennlp evaluate  $new save_directory$ \
  $test_data$ \
  --include-package $model_file$ \
  --output-file $output_directory$
```

