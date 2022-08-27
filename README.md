# Code for LAB-Net
## Paper name
LAB-Net: A Lightweight Network Based on LAB Color Space for Shadow Removal
## Requirements
python=3.7.13

pytorch=1.12.1
```bash
pip install -r requirments.txt
```
## Train
### Train ISTD
#### 1. Modify './script/train.sh'
1. Adjust loadSize(256), FineSize(256), down_w(256), down_h(256)
2.  Add dataroot(ISTD trainset path), name(task name)
#### 2. Run 'train.sh'
```bash
cd script
bash train.sh 0
```
0 is the gpu number
### Train SRD
#### 1. Modify './script/train.sh'
1. Adjust batchs(1)
2. Adjust loadSize(400), FineSize(400), down_w(128), down_h(128)
3.  Add dataroot(SRD trainset path), name(task name)
#### 2. Run 'train.sh'
```bash
cd script
bash train.sh 0
```
0 is the gpu number
### See loss
You can see the train loss:
```bash
cd script
tensorboard --logdir LAB_G_LABNet_name
```
## Test
1. You can download our pretrained model to test.

Our ISTD checkpoint can be found [here](https://drive.google.com/drive/folders/1GWLLBi-ZBREnqWPSCgCv8Ha-8iiLILXS?usp=sharing)

Our SRD checkpoint can be found [here](https://drive.google.com/drive/folders/15jwF-Sq3xFWJL_tGorhRsnSr5ya9GITX?usp=sharing)

Please move the .pth to a directory to use.
### Test ISTD
#### 1. Modify './script/test.sh'
1. Adjust size_w(640), size_h(480), down_w(256), down_h(256)
2. Add dataroot(ISTD testset path), name(task name), resroot(path to save the result)
#### 2. Run 'test.sh'
```bash
cd script
bash test.sh 0
```
0 is the gpu number
### Test SRD
#### 1. Modify './script/test.sh'
1. Adjust size_w(840), size_h(640), down_w(128), down_h(128)
2. Add dataroot(SRD testset path), name(task name), resroot(path to save the result)
#### 2. Run 'test.sh'
```bash
cd script
bash test.sh 0
```
0 is the gpu number
## Our result
### ISTD
<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow" colspan="3">S</th>
    <th class="tg-c3ow" colspan="3">NS</th>
    <th class="tg-c3ow" colspan="3">ALL</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-c3ow">RMSE</td>
    <td class="tg-c3ow">PSNR</td>
    <td class="tg-c3ow">SSIM</td>
    <td class="tg-c3ow">RMSE</td>
    <td class="tg-c3ow">PSNR</td>
    <td class="tg-c3ow">SSIM</td>
    <td class="tg-c3ow">RMSE</td>
    <td class="tg-c3ow">PSNR</td>
    <td class="tg-c3ow">SSIM</td>
  </tr>
  <tr>
    <td class="tg-c3ow">6.65</td>
    <td class="tg-c3ow">37.17</td>
    <td class="tg-c3ow">0.9887</td>
    <td class="tg-c3ow">4.49</td>
    <td class="tg-c3ow">32.42</td>
    <td class="tg-c3ow">0.9727</td>
    <td class="tg-c3ow">4.84</td>
    <td class="tg-c3ow">30.49</td>
    <td class="tg-c3ow">0.9563</td>
  </tr>
</tbody>
</table>

Our ISTD result can be found [here](https://drive.google.com/drive/folders/12IO_H3uOynFXshx4K4gOWea2tATXKct3?usp=sharing)

### SRD
<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow" colspan="3">S</th>
    <th class="tg-c3ow" colspan="3">NS</th>
    <th class="tg-c3ow" colspan="3">ALL</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-c3ow">RMSE</td>
    <td class="tg-c3ow">PSNR</td>
    <td class="tg-c3ow">SSIM</td>
    <td class="tg-c3ow">RMSE</td>
    <td class="tg-c3ow">PSNR</td>
    <td class="tg-c3ow">SSIM</td>
    <td class="tg-c3ow">RMSE</td>
    <td class="tg-c3ow">PSNR</td>
    <td class="tg-c3ow">SSIM</td>
  </tr>
  <tr>
    <td class="tg-c3ow">6.56</td>
    <td class="tg-c3ow">35.71</td>
    <td class="tg-c3ow">0.9818</td>
    <td class="tg-c3ow">3.77</td>
    <td class="tg-c3ow">36.5</td>
    <td class="tg-c3ow">0.9813</td>
    <td class="tg-c3ow">4.6</td>
    <td class="tg-c3ow">32.22</td>
    <td class="tg-c3ow">0.9554</td>
  </tr>
</tbody>
</table>

Our SRD result can be found [here](https://drive.google.com/drive/folders/1G3oWIYnk2EYxl3t1-aLVGoKExGXM0car?usp=sharing)
