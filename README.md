## Introduction  
**STHGCN-Traffic-Net** is a studing project based on https://github.com/ChyaZhang/ChatTraffic and https://github.com/yysijie/st-gcn Two Articles and Codes.

**STHGCN-Traffic-Net** is a novel model based on Spatio-Temporal Hypergraph Convolutional Networks, developed to improve the accuracy and efficiency of traffic accident prediction. By combining spatio-temporal modeling with hypergraph structures, this model captures complex multi-node relationships and temporal dependencies within traffic networks. It is specifically tailored for predicting traffic-related incidents, leveraging both structural and temporal data from real-world datasets. The project builds upon key methodologies and codebases from existing works to further the understanding and application of advanced graph-based neural networks in traffic analysis.  

## Requirements  
The code for this project is based on the [ChatTraffic](https://github.com/ChyaZhang/ChatTraffic) repository, which itself integrates techniques from latent diffusion models. To set up the environment and get started with the codebase, follow the instructions below:  

```bash  
git clone https://github.com/ChyaZhang/ChatTraffic.git  
cd ChatTraffic  
conda env create -f environment.yaml  
conda activate ChatTraffic  
```  

Ensure that you have the necessary dependencies installed, including Python, PyTorch, and other required libraries specified in the `environment.yaml` file.  

## Dataset  
The model uses the **Beijing Text-Traffic (BjTT)** dataset, a comprehensive collection of traffic data designed for predictive modeling. The dataset includes spatio-temporal traffic features and related annotations, which are crucial for the performance of the model. Please note that access to the BjTT dataset requires permission from the dataset creators. Users interested in using this dataset should request permission directly.  

## Result  
The output of the model is generated in `.npy` format, which stores the predicted traffic data. To visualize these predictions, you can use the provided `scripts/plot_map.py` script. This script allows you to display the processed traffic data on a map, enabling a clear and intuitive understanding of the prediction results.  

To visualize the data:  
1. Run the model and save the output in `.npy` format.  
2. Use the script:  
   ```bash  
   python scripts/plot_map.py --input_path path_to_output_file.npy  
   ```  
3. The traffic data will be rendered on a map for detailed analysis and evaluation.  

## Future Work  
Further development of **STHGCN-Traffic-Net** will focus on:  
- Expanding its application to other urban traffic datasets.  
- Optimizing hyperparameters for better prediction accuracy.  
- Exploring the integration of additional real-time traffic features for enhanced robustness.

## Important
In this project. We don't have other useage rather than learning.

