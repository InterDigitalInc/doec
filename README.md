 # Octree Entropy Coding

This repository contains the implementation of "Point cloud geometry compression using learned octree entropy coding" (m59528 & m59529) by InterDigital. It is implemented based on the [pccAI](https://github.com/InterDigitalInc/pccAI) (*pick-kai*) framework—a PyTorch-based framework for conducting AI-based Point Cloud Compression (PCC) experiments.

## Installation

We tested our implementation on Python 3.6, PyTorch 1.7.0 and CUDA 10.1, under a conda virtual environment. For installation, please launch our installation script `install_torch-1.7.0+cu-10.1.sh` with the following command:
```bash
echo y | conda create -n oec python=3.6 && conda activate oec && ./install_torch-1.7.0+cu-10.1.sh
```
It is highly recommended to look at the installation script which describes the details of the necessary packages. After that, put the binary of `pc_error` (MPEG D1 & D2 computation) under the `third_party` folder.

## Datasets
Create a `datasets` folder then put all the datasets below. One may create soft links to the existing datasets to save space.
### Ford Sequences

We use the first *Ford* sequences for training and the other two sequences for benchmarking, arranged as follows:
```bash
${ROOT_OF_THE_REPO}/datasets/ford
                               ├── ford_01_q1mm
                               ├── ford_02_q1mm
                               └── ford_03_q1mm
                                       ├── Ford_03_vox1mm-0200.ply
                                       ├── Ford_03_vox1mm-0201.ply
                                       ├── Ford_03_vox1mm-0202.ply
                                       ...
                                       └── Ford_03_vox1mm-1699.ply
```

## Basic Usages

The core of the training and benchmarking code are below the `pccai/pipelines` folder. They are called by their wrappers below the `experiments` folder. The basic way to launch experiments with pccAI is:
 ```bash
 ./scripts/run.sh ./scripts/[filename].sh [launcher] [GPU ID(s)]
 ```
where `launcher` can be `s` (slurm), `d` (direct, run in background) and `f` (direct, run in foreground). `GPU ID(s)` can be ignored when launched with slurm. The results (checkpoints, point cloud files, log, *etc.*) will be generated under the `results/[filename]` folder. Note that multi-GPU training/benchmarking is not supported for training networks using sparse convolutions (i.e., SparseVCN).

 ### Benchmarking

One can use the following command lines for benchmarking the selected rate points individually, followed by merging the generated CSV files for MPEG reporting:
 ```bash
for i in {1..4}
do
   ./scripts/run.sh ./scripts/oec/bench_ford_pcn_r0$i.sh f 0
done
python ./utils/merge_csv.py --input_files ./results/bench_ford_pcn_r01/mpeg_report.csv ./results/bench_ford_pcn_r02/mpeg_report.csv ./results/bench_ford_pcn_r03/mpeg_report.csv ./results/bench_ford_pcn_r04/mpeg_report.csv --output_file ./results/bench_ford_pcn/mpeg_report.csv
 ```

BD metrics and R-D curves are generated via the [MPEG reporting template for AI-based PCC](http://mpegx.int-evry.fr/software/MPEG/PCC/ai/mpeg-pcc-ai-report) (also available publically via [GitHub](https://github.com/yydlmzyz/AI-PCC-Reporting-Template)). For example, run the following command right under the folder of its repository:
```bash
python test.py --csvdir1='csvfiles/reporting_template_lossy.csv' --csvdir2='/PATH/TO/mpeg_report.csv' --csvdir_stats='csvfiles/reporting_template_stats.csv' --xlabel='bppGeo' --ylabel='d1T'
```
It can also generate the average results for a certain category:
```bash
python test_mean.py --category='am_frame' --csvdir1='csvfiles/reporting_template_lossy.csv' --csvdir2='/PATH/TO/mpeg_report.csv' --csvdir_stats='csvfiles/reporting_template_stats.csv' --xlabel='bppGeo' --ylabel='d1T'
```

Replace `d1T` with `d2T` for computing the D2 metrics. The benchmarking of surface point clouds can be done in the same way. All the scripts for benchmarking are put under the `scripts/oec` folder. Please refer to the related MPEG contributions for example R-D curves.

### Training

Take the training of the Ford sequences as example, one can directly run
 ```bash
./scripts/run.sh ./scripts/oec/train_ford_pcn.sh d 0
 ```
which trains the deep entropy model for Ford sequences. The trained model will be generated under the `results/train_ford_pcn` folder.

To understand the meanings of the options in the scripts for benchmarking/training, refer to `pccai/utils/option_handler.py` for details.

NOTE: Training and benchmarking scripts are also provided for VCN and SparseVCN.

## License
OEC code is released under the BSD License, see `LICENSE` for details.

## Contacts
Please contact Muhammad Lodhi (muhammad.lodhi@interdigital.com), for any questions.

## Related Resources
 * [pccAI](https://github.com/InterDigitalInc/pccAI)
 * [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine)
 * [mpeg-pcc-ai-report](http://mpegx.int-evry.fr/software/MPEG/PCC/ai/mpeg-pcc-ai-report) / [AI-PCC-Reporting-Template](https://github.com/yydlmzyz/AI-PCC-Reporting-Template)
 * [TMC13](https://github.com/MPEGGroup/mpeg-pcc-tmc13)
 * [DeepZip](https://github.com/mohit1997/DeepZip/blob/master/src/arithmeticcoding_fast.py) (used for arithemtic coding)
 * [PointNet++](https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet2_utils.py) (used for set abstraction module in PointContextNet)
 