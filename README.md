## Dependencies
- numpy==1.26.1
- torch==2.2.1  
- torch-geometric==2.5.2  
- torch-cluster==1.6.3  
- torch-sparse==0.6.18   
- torch-scatter==2.1.2  
- transformers==4.48.2 
- datasets==3.3.2
- networkx==3.2.1
- python-louvain==0.16


## Usage
##### 1. Install dependencies
```
conda create --name FederatedTAG -y python=3.9.18
conda activate FederatedTAG
pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.26.1 torch-geometric==2.5.2 transformers==4.48.2 datasets==3.3.2 networkx==3.2.1
pip install torch-cluster torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.2.1+cu121.html
```

##### 2. Run code
Run *FedAvg* on *History* with *BERT & GPT-2* as the PLMs
```
python fedavg.py --dataset_name=History --plm=bert+gpt2 
```

## Parameters

| Parameter    | Description                                                                         | 
|--------------|-------------------------------------------------------------------------------------|
| dataset_name | Dataset to use. Options: [History, Photo, Children, Computers, Fitness]             |
| data_path    | Folders storing the original data (default: '/mnt/nvme0/xingbo/FederatedTAG/CSTAG') |
| hidden       | hidden size of GNN (default: 128)                                                   |
| plm          | PLMs. Options: bert+gpt2, bert+roberta, gpt2+roberta                                |
| lr           | learning rate (default: 0.005)                                                      |
| epochs       | epochs (default: 5)                                                                 |
| rounds       | number of rounds (default: 500)                                                     |
| gpu_id       | the GPU that will be used (default: 0)                                              |

