<h2 align="center">IEMask R-CNN: Information-enhanced Mask R-CNN</h2>
<h4 align="right">by <a href="http://faculty.cqupt.edu.cn/bixiuli/zh_CN/index.htm">Xiuli Bi</a>, <a href="https://fhujinwu.github.io/">Jinwu Hu</a>, <a href="https://faculty.cqupt.edu.cn/xiaobin/zh_CN/index.htm">Bin Xiao*</a>, <a href="https://faculty.cqupt.edu.cn/liws/zh_CN/index.htm">Weisheng Li</a>, <a href="https://see.xidian.edu.cn/faculty/xbgao/">Xinbo Gao</a></h4>

<div align="center">
  <img src="./images/fig2.PNG"><br><br>
</div>
<div align="center">
  <img src="./images/table5.png"><br><br>
</div>

This is an official implementation of IEMask R-CNN in our IEEE Transactions on Big Data paper "
<a href="https://ieeexplore.ieee.org/document/9811396">
IEMask R-CNN: Information-enhanced Mask R-CNN</a>"

## Usage
### Note
Our code is based on the <a href="https://github.com/facebookresearch/detectron2">
Detectron2</a> and <a href="https://github.com/hustvl/BMaskR-CNN">BMask R-CNN</a> implementation.

### Training

```bash
python tools/train_net.py --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml --num-gpus 1
```

### Evaluation

specify a config file and test with trained model

```bash
python train_net.py --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS ./output/model_final.pth
```


## Citation
If you use IEMask R-CNN in your research, please cite our IEEE Transactions on Big Data paper.

```text
@ARTICLE{9811396,
  author={Bi, Xiuli and Hu, Jinwu and Xiao, Bin and Li, Weisheng and Gao, Xinbo},
  journal={IEEE Transactions on Big Data}, 
  title={IEMask R-CNN: Information-Enhanced Mask R-CNN}, 
  year={2023},
  volume={9},
  number={2},
  pages={688-700},
  doi={10.1109/TBDATA.2022.3187413}}
```
 
## update status
The code (V1) is uploaded (Ongoing updates).

