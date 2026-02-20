### Towards Robust Neuromorphic/Event Vision-Based Fall Detection: Cross-Dataset Generalisation Benchmarking"


## Introduction
This project studies **event-camera-based fall detection** under realistic domain shifts. We train fall detectors on **synthetic event streams** generated from multiple RGB fall datasets and validate generalization on a **native neuromorphic dataset (PAF)**. The goal is to stress-test robustness across **different recording conditions, subjects, and event-generation methods**, not just optimize in-dataset accuracy. See **Table II** for the full dataset overview.

## Datasets
We train and evaluate our fall detection approach using **four RGB datasets converted to event-camera format**, with validation on the **native PAF neuromorphic dataset** (see **Table II**). The RGB datasets are converted to event-stream format using the **Metavision Video-to-Event (V2E)** converter with standard DVS parameters, enabling synthetic event data generation from diverse RGB sources.

### RGB-to-Event (Synthetic Event Streams)
- **GMDCSA-24**  
  A compact dataset with **160 sequences** (**81 falls**, **79 ADL**) from **4 subjects**, including **frame-level temporal annotations**. Its small footprint (~**1.95 GB**) makes it a practical choice for **edge deployment** and rapid iteration. 

- **Multiple Cameras**  
  Includes **24 calibrated multi-camera scenarios** with **9 position states** and **explicit temporal phase annotations** that support **fall decomposition** (i.e., breaking a fall into phases).

- **CAUCAFall**  
  Contains **100 videos** with **20,002 per-frame binary annotations** from **10 subjects**, recorded in **uncontrolled environments**. It captures real-world nuisance factors like **variable lighting** and **occlusions**, making it valuable for robustness testing.

- **FallVision**  
  The largest dataset used here, comprising **11,732 videos** from **58 subjects**. It provides detailed **17-point pose keypoint annotations** across **multiple recording angles and positions**, supporting richer supervision and analysis.

### Native Event Dataset (Real Neuromorphic Recordings)
- **PAF (neuromorphic)**  
  Used for cross-dataset generalization evaluation after training on RGB-to-event data. PAF includes **180 native neuromorphic recordings** captured with the **DAVIS346redColor event camera** from **15 subjects** performing four action classes: **falling**, **bending**, **slumping**, and **tying shoes**.

  Link: https://github.com/CrystalMiaoshu/PAFBenchmark

## RGB to Event Conversion
We have used Metavision SDK Video to Event Simulator. This allows you to transform frame-based image or video into event-based counterpart.

Link: https://docs.prophesee.ai/stable/samples/modules/core_ml/viz_video_to_event_simulator.html

## Synthetic Event Datasets
The RGB to Event Converted Datasets are publicaly available:

E-GMDCSA-24: https://dataverse.no/dataset.xhtml?persistentId=doi:10.18710/T0GJXA

E-CAUCAFall: https://dataverse.no/dataset.xhtml?persistentId=doi:10.18710/ITNCWP

E-FallVision: https://dataverse.no/dataset.xhtml?persistentId=doi:10.18710/SJG4E6

E-MCFA: The usage license for original dataset is not defined by the authors. We are currently acquiring the license information as soon as we get the license and permision, we will upload this dataset.


## Cross-Dataset Evaluation Protocol
Following training on the converted RGB-to-event datasets, we evaluate model generalization on **PAF**. This strategy—combining **synthetic event data** from diverse RGB sources with **native event-camera recordings**—provides a rigorous assessment of **robustness** and **domain adaptation** across different event generation methods and sensing conditions.

