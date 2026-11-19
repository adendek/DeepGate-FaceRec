# DeepGate-FaceRec — Dataset Documentation

This directory contains face dataset used for the **DeepGate-FaceRec** homework assignment.
It is derived from the **Labeled Faces in the Wild (LFW)** benchmark, available on [Kaggle](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset).

---

## 1. Dataset Overview & Curation Logic

In biometric metric learning and face verification, standard random image splitting causes **severe data leakage**.

To prevent this, the dataset is strictly **identity-disjoint**:
$$\text{Identities}(\text{Train}) \cap \text{Identities}(\text{Val}) \cap \text{Identities}(\text{Test}) = \emptyset$$

### Filtering Criteria
The raw LFW corpus contains 5,749 identities, but the majority have only a single image. We filtered the dataset using the following rules:
- **Minimum images per identity ($> 4$):** Ensures that every identity can form valid anchor-positive $(a, p)$ pairs during $P \times K$ batch construction.
- **Maximum images per identity ($< 100$):** Excludes dominant public figures (e.g., George W. Bush with 500+ photos) to avoid severe class imbalance and batch dominance during triplet mining.

---

## 2. Obtaining the Dataset

To download and prepare the dataset, run the following command from the main project folder:

```bash
uv run python `scripts/download_data.py`
```

The script downloads a single ZIP archive (~45 MB) and extracts the identity-disjoint train/, val/, and test/ splits.

Please confirm that the zip hash is

```
sha256:6bd700f9eebe5dad3561bcad79fd949b6db25b3759974faef16d54fe5f1cdefd
```

## 3. Directory Layout

The prepared dataset is organized under `data/split/` as follows:

```text
data/
├── README.md                     # This file
└── split/
    ├── train_identities.csv      # List of identities assigned to training
    ├── val_identities.csv        # List of identities assigned to validation
    ├── test_identities.csv       # List of identities assigned to testing
    │
    ├── train/                    # ~70% of identities (Metric representation learning)
    │   ├── Colin_Powell/
    │   │   ├── Colin_Powell_0001.jpg
    │   │   └── Colin_Powell_0002.jpg
    │   └── Tony_Blair/
    │       └── ...
    │
    ├── val/                      # ~15% of identities (Open-set validation monitoring)
    │   ├── Gerhard_Schroeder/
    │   └── ...
    │
    └── test/                     # ~15% of identities (Simulated Smart Access Gate)
        ├── gate_gallery/         # Enrolled reference database (1 reference photo per staff member)
        │   ├── Staff_PersonA_0001.jpg
        │   └── Staff_PersonB_0001.jpg
        │
        └── gate_probes/          # Access attempts at the smart gate
            ├── authorized/       # Unseen query photos of enrolled staff (Ground Truth: UNLOCK)
            │   ├── Staff_PersonA_0002.jpg
            │   └── Staff_PersonB_0002.jpg
            └── impostors/        # Query photos of un-enrolled strangers (Ground Truth: DENY)
                ├── Impostor_PersonX_0001.jpg
                └── Impostor_PersonY_0001.jpg
```

## 4. Dataset Limitations, Demographic Bias & Intended Use

- **Toy Dataset for Educational Use Only:** This curated subset is a small **toy dataset** intended strictly for learning deep metric learning, triplet mining, and threshold calibration (`SmartGate`). Models trained on this data are for coursework exercises only and must **not** be used in production biometric or access-control systems.
- **Demographic Bias:** Because LFW was compiled from Western news photography in the early 2000s, it exhibits severe demographic skew. In particular, it is heavily biased toward **white individuals** (alongside male and older public figures).
- **Fairness Implications:** Training face recognition models on demographically homogeneous datasets leads to degraded embedding separation and uneven error rates (FAR/FRR) across underrepresented racial, gender, and age groups.
- **Balanced Alternatives & Further Reading:** For research and benchmarks specifically designed to address demographic skew in face datasets, see Kärkkäinen and Joo (WACV 2021), *[FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age](https://openaccess.thecvf.com/content/WACV2021/papers/Karkkainen_FairFace_Face_Attribute_Dataset_for_Balanced_Race_Gender_and_Age_WACV_2021_paper.pdf)*.
