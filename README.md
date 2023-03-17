# Mineral-Identification

This repo provides the dataset and code for reproducing the experiments in paper ---- [Mineral Identification Based on Deep Learning That Combines Image and Mohs Hardness](https://www.mdpi.com/2075-163X/11/5/506/htm).



## Data

* We crawled 36 species of mineral images from the [mindat](https://www.mindat.org) website, a total of 183,688 images. 
* The specific species are as follows: Agate, Albite, Almandine, Anglesite, Azurite, Beryl, Cassiterite, Chalcopyrite, Cinnabar, Copper, Demantoid, Diopside, Elbaite, Epidote, Fluorite, Galena, Gold, Halite, Hematite, Magnetite, Malachite, Marcasite, Opal, Orpiment, Pyrite, Quartz, Rhodochrosite, Ruby, Sapphire, Schorl, Sphalerite, Stibnite, Sulphur, Topaz, Torbernite, Wulfenite.



## How to run code

#### Dependency

* pytorch

#### Run

* If you have only one GPU, you can run `EfficientNet_36classes_trainABatch_ValABatch.py` and `EfficientNet_hardness_36classes_trainABatch_ValABatch.py` directly.

  ```bash
  # in bash
  python EfficientNet_36classes_trainABatch_ValABatch.py
  ```

  ```bash
  # in bash
  python EfficientNet_hardness_36classes_trainABatch_ValABatch.py
  ```

* If you have more than one GPU, you can use distributed training.

  ```bash
  # in bash
  # nproc_per_node is your GPU nums
  python -m torch.distributed.launch --nproc_per_node=2 EfficientNet_36classes_trainABatch_ValABatch.py
  ```

  ```bash
  # in bash
  # nproc_per_node is your GPU nums
  python -m torch.distributed.launch --nproc_per_node=2 EfficientNet_hardness_36classes_trainABatch_ValABatch.py
  ```

  

## Application

* An app on smartphones using our trained model is implemented.

* For the Android or ios version, you can search "mineral identification" in Google Play or Apple's app store. And you can also visit link [Android version](https://play.google.com/store/apps/details?id=com.kate.study3) and [ios version](https://apps.apple.com/cn/app/mineral-identification/id1537377326) directly.
