# UniOCTSeg++

Official repository for **UniOCTSeg++: Refined Hierarchical Prompt Strategy and Bi-directional Progressive Consistency Learning for Universal Retinal Layer Segmentation in OCT**.

This repository provides:
- the official implementation of **UniOCTSeg++**
- access instructions for the **HROCT-Bench** dataset

---

## Code

The official implementation of **UniOCTSeg++** is released in this repository.

---

## HROCT-Bench Dataset Access

We provide access instructions for **HROCT-Bench**, a unified benchmark for universal retinal layer segmentation in OCT.

### 📂 Publicly Available Subsets

The following subsets do **not require additional permission** from the original data providers and are released in **preprocessed form**:

- HC-MS
- A2A-SDOCT
- DUKE DME
- DUKE AMD
- HEG
- OIMHS

👉 Hugging Face: [HROCT-Bench](https://huggingface.co/datasets/Hal1010/HROCT-Bench)  
👉 Baidu Netdisk: [百度网盘](https://pan.baidu.com/s/1vY5fbcF3GpnYl1_lfbFF0A?pwd=gyi4)  
**Password:** `gyi4`

---

### 🔒 Restricted Subsets

The following subsets are **not redistributed in this repository** due to access restrictions imposed by the original data providers:

- **OCTA-500**: https://ieee-dataport.org/open-access/octa-500  
- **GCN**: https://yuyeling.com/project/mgu-net/  
- **GOALS**: https://aistudio.baidu.com/competition  
- **NR206**: https://github.com/Medical-Image-Analysis/NR206-Dataset/tree/main/dataset  
- **Harvard-EF30K**: https://github.com/osamakhaan/Harvard-EyeFairness  

To use these subsets in **HROCT-Bench**, users should:

1. Obtain the raw data directly from the original dataset providers via the links above, in accordance with their corresponding access policies and license terms.
2. Use the preprocessing scripts provided in this repository to process the original data.
3. Reproduce the preprocessed format used in **HROCT-Bench** for training and evaluation.

---

## License and Data Usage

### Code License

The code in this repository is released under the license specified in this repository.

### Data License

The **HROCT-Bench** data released here are curated and preprocessed from multiple publicly accessible source datasets. The use of each subset remains subject to the **license terms, access conditions, and redistribution policies of the corresponding original dataset provider**.

For the publicly released subsets, we only share the preprocessed data that are allowed to be redistributed.  
For the restricted subsets, we do **not** redistribute either the raw data or the preprocessed data. Instead, we provide the original access links and the corresponding preprocessing scripts, so that users can obtain the raw data from the original sources and reproduce the benchmark format themselves.

By using this repository and the associated data resources, users are responsible for complying with the terms of the original datasets. This repository does **not** claim to relicense the underlying raw data beyond what is permitted by the original data providers.

---

## 📌 Notes

- All released data follow the preprocessing pipeline described in our paper.
- This repository includes the official implementation of **UniOCTSeg++** as well as the released subsets and access instructions for **HROCT-Bench**.
- These resources are provided to support transparency, reproducibility, and future research on universal retinal OCT layer segmentation.
- For any questions regarding data access, please contact: **1774706797z@gmail.com**

---

## Citation

If you find this repository useful, please consider citing our paper.
