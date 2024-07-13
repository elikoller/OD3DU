<div align='center'>
<h2 align="center"> SceneGraphUpdate Bachelorthesis of Elena </h2>

<a href="https://y9miao.github.io/">Elena Koller</a><sup>1</sup>, 
<a href="https://cvg.ethz.ch/team/Dr-Francis-Engelmann">Zuria Bauer</a><sup>1</sup> , 
<a href="https://cvg.ethz.ch/team/Dr-Daniel-Bela-Barath"> Dániel Béla Baráth</a> <sup>1</sup>

<sup>1</sup>ETH Zurich   

I don't knwo what to write here but good template so to do: overwrite his stuff this is only a buffer right now


![teaser](./repo_info/TeaserImage.jpg)
</div>




## News :newspaper:

* **26. Mar 2024**: Code released.

## Code Structure :clapper:

```
├── BT
│   ├── preprocessing         <- data preprocessing
│   ├── configs               <- configuration definition
│   ├── src
│   │   │── datasets          <- dataloader for 3RScan and Scannet data
│   │   
│   ├── scripts               <- implementation scripts 
│   │── utils                 <- util functions
│   │── README.md                    
```

### Dependencies :memo:

The project has been tested on Ubuntu 20.04.
The main dependencies of the project are the following:



```yaml
python: 3.8.15
cuda: 11.6
```
You can set up an environment as follows :
```bash
git clone https://github.com/y9miao/VLSG.git
cd VLSG

conda create -n "BT" python=3.8.15
conda activate BT
pip install -r requirement.txt
```
Other dependences:

also we need python -m pip install pyviz3d for the visualization 

Some dependencies are useless and give errors: cat requirement.txt | xargs -n 1 pip install

this installs them and skipps whenever something is not working

also installed


the thing below is nt needed
```bash
conda activate VLSG
pip install -r other_deps.txt

cd thrid_party/Point-NN
pip install pointnet2_ops_lib/.
```

## Dataset Generation :hammer:
### Download Dataset - 3RScan + 3DSSG
Download [3RScan](https://github.com/WaldJohannaU/3RScan) and [3DSSG](https://3dssg.github.io/). Move all R3Scan files to ``3RScan/scenes/``, all files of 3DSSG to a new ``3RScan/files/`` directory within Scan3R. The additional meta files are available [here](https://drive.google.com/file/d/1abvycfnwZFBBqYuZN5WFJ80JAB1GwWPN/view?usp=sharing). Download the additional meta files and move them to ``3RScan/files/``.
The structure should be:

```
├── 3RScan
│   ├── files                 <- all 3RScan and 3DSSG meta files and annotations
│   │   ├──Features2D         <- Pre-computed patches features of query images
│   │   ├──Features3D         <- Visual features of 3D objects not yet
│   │   ├──orig               <- Scene Graph Data
│   │   ├──patch_anno         <- Ground truth patch-object annotation of query images
│   │   meta files
│   ├── scenes                <- scans
```

> To generate ``labels.instances.align.annotated.v2.ply`` for each 3RScan scan, please refer to the repo from 
[here](``https://github.com/ShunChengWu/3DSSG/blob/master/data_processing/transform_ply.py``).  https://github.com/ShunChengWu/3DSSG/blob/main/data_processing/transform_ply.py

(to do find out how that works because I got the link with the files from yang but the link expired :(      )

To unzip the sequence files within the §RScan/scenes/  you can use
directly in the terminal



cd /local/home/ekoller/R3Scan/


for folder in scenes/*/; do


    (cd "$folder" && unzip -o '*.zip' -d sequence && rm -f *.zip)
    
done



also some more confusing stuff is the following: the dataset did not provide the graphdata for the evaluation set so in this paper: the original validation set was taken and then split into validation and test. hence for the dataset only the test and validation set get accessed


### Dataset Pre-process :hammer:
After installing the dependencies, we download and pre-process the datasets. 

First, we pre-process the scene graph information provided in the 3RScan annotation. The relevant code can be found in the ``data-preprocessing/`` 
directory.  

E: Some changes happened I think the directory data-preprocessing is now called preprocessing only, also adjustment of the Data_Root_dir to the new one (plus the adding of the definition into the utis scan3r.py that you sent me)

Oh also there is Data root dir in a lot of graphs below 

Don't forget to set the env variables "VLSG_SPACE" as the repository path,  set "Data_ROOT_DIR" as the path to "3RScan" dataset and set "CONDA_BIN" to accordingly in the bash script.

```bash
bash scripts/preprocess/scan3r_data_preprocess.sh
```
The result processed data will be save to "{Data_ROOT_DIR}/files/orig".
<!-- > __Note__ To adhere to our evaluation procedure, please do not change the seed value in the files in ``configs/`` directory.  -->

### Generating Ground Truth Patch-Object Annotastion
To generate ground truth annotation, use : 
```bash
bash scripts/gt_annotations/scan3r_gt_annotations.sh
```
This will create a pixel-wise and patch-level ground truth annotations for each query image. These files will be saved  to "{Data_ROOT_DIR}/files/gt_projection and "{Data_ROOT_DIR}/files/patch_anno only for the eval and train set tho
this also directly computes the boundingboxes for the objects in the projections for every scan

while it migh be confusing the ground truth for the reference is actually the data we already have provided: our assumption is that we have the mesh -> so there for the reference scans we will actually compute with them. for the rescans however this is the actual ground truth which we will also use for comparison later on




### Elena's Code
In the preprocessing added the calculation of the boundingboxes. For this there is also a file in the preprocessing calles boundingboxVisuals. In this notebook everything about the boundingbox generation can be found and also some visualizations. The modifications added to the preprocess_scan3r.py file are clearly marked as modified by me in case of need for change.

open questions: why tf are there differend amounts of ids saved when accessind the data differently -> texted yang wait for answer

Rayshooting: in the subsection of preprocessing/ray_shooting_pixel_wise_ray_shooting there is a notebook which shows the calculation steps for the calculation of the rays. it dived into the ray generation, and intersection with the boundingboxes, returning not only the intersection point but also the id of the boundingbox which got intersected. Alto the visualizations of the rays are open

open to dos: 
- to correct the intersection stuff start with reference scan -> the boundingboxes should hit
- visualize the mesh: there I had to change the rgb to bgr or something wtf? What was it before??? What does this mean for the rest :((((((
- also plot the boundingboxes in three- looks weird form angle but align

- also plot the rays in there -> now we got the porblem:: wtf look again at matricies or maybe some issue with meter conversion but else same structure as Yangs intersection so wtf

- very slow -> intersection with the boundingboxes can use some speedup work on that
- when the comparison with the gt is not the same: show the transformation in a semantic way: what became what
plot the depht map to see the dimensions ( not highest prio atm but do it)
- check if the Id are correctly taken!: should be correctly taken. according to FAQ id in object.jason is the same as objdctId or Id in semseg.json

### Generating Boundingboxes for the input rgb using 
the bigger goal is to generate dinov features for different objects in the input images and matching them to the objects in the scene for this we use sam segment anything to compute boundinboxes for the objects. the boundingboxes get extracted and saved into "{Data_ROOT_DIR}/files/sam_data
```bash
bash scripts/sam_data/sam_segmentations.sh
```

### Patch-Level Features Pre-compute
In order to speed up training, we pre-compute the patch-level features with 
      [Dino v2](https://dinov2.metademolab.com/). 
To generate the features, use : 
```bash
bash scripts/features2D/scan3r_dinov2.sh
```
This will create patch-level features for query images and save in "{Data_ROOT_DIR}/Features2D/DinoV2_16_9_scan".  





## BibTeX :pray:
```
@misc{miao2024scenegraphloc,
      title={SceneGraphLoc: Cross-Modal Coarse Visual Localization on 3D Scene Graphs}, 
      author={Yang Miao and Francis Engelmann and Olga Vysotska and Federico Tombari and Marc Pollefeys and Dániel Béla Baráth},
      year={2024},
      eprint={2404.00469},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
 ```

## Acknowledgments :recycle:
In this project we use (parts of) the official implementations of the following works and thank the respective authors for sharing the code of their methods: 
- [SGAligner](https://github.com/sayands/sgaligner) 
- [OpenMask3D](https://openmask3d.github.io/)
- [Lip-Loc](https://liploc.shubodhs.ai/) 
- [Lidar-Clip](https://github.com/atonderski/lidarclip)
- [AnyLoc](https://github.com/AnyLoc/AnyLoc)
- [CVNet](https://github.com/sungonce/CVNet)
- [SceneGraphFusion](https://github.com/ShunChengWu/3DSSG)

