<div align="center">

# Pathology foundation model embeddings as a microenvironment representation for counterfactual gene expression prediction

### Lab rotation report

**David Špiljak**<sup>1,2</sup>

Supervisor: Daniel Dimitrov<sup>1,3</sup>

<sup>1</sup>European Molecular Biology Laboratory (EMBL), Heidelberg, Germany<br>
<sup>2</sup>Heidelberg University, Heidelberg, Germany<br>
<sup>3</sup>German Cancer Research Center (DKFZ), Heidelberg, Germany

9 October 2026

</div>

<br>

<div align="center"><b>Abstract</b></div>

> Spatial transcriptomics (ST) resolves gene expression in its tissue context, but its cost still limits clinical use, while images of haematoxylin and eosin (H&E) are cheap and available routinely. Cellina recently showed that cell expression can be disentangled into an intrinsic component and an extrinsic microenvironment-driven component, which makes niche counterfactuals possible. Here we ask whether embeddings from pathology foundation models can stand in for the microenvironment representation that Cellina learns from ST, so that such counterfactuals could in principle be posed from H&E alone. Using a Xenium Prime section of human ovarian cancer with a matched H&E image for every cell, we first show that a simple linear classifier on cell-centred tile embeddings recovers the tissue domain almost perfectly, while it recovers the cell type poorly. We then build a decoder-only analogue of Cellina that predicts expression from a cell-type label combined with either a domain label or an H&E embedding, and use it to move Stromal cells from Smooth muscle into a Tumor context *in silico*. On held-out Tumor regions, the image-conditioned model approaches the dedicated H&E-to-expression predictor Path2Space and matches it at the level of whole cell populations, despite being trained on a single slide with a small decoder. Image embeddings therefore capture the microenvironment well; representing the cell's intrinsic state from H&E remains the main open problem.

<br>

---

## 1 Introduction

Spatial omics has reshaped frontier biological research since it became widely available to the scientific community. A whole new field has formed around it, answering questions previously considered out of reach because of experimental limitations; quantifying cell–cell interactions or the movement tendencies of cells within a tissue, for example, has become fairly routine. This makes spatial omics a promising route to informed decisions in cancer treatment, where tissue morphology, tumour heterogeneity and cell states are essential to understand before crafting a personalized treatment for the patient ([Lewis et al., 2021](#ref-lewis_spatial_2021); [Bressan et al., 2023](#ref-bressan_dawn_2023)). Yet the technology is still too expensive for widespread clinical use, so clinicians still rely mainly on pathology images such as H&E scans to assess tumour heterogeneity and morphology, thanks to their reliability and low cost.

Conveniently, most ST experiments already produce an H&E image of the same tissue section, or of a neighbouring one, so paired morphology and expression data come as a by-product of the experiment ([Jaume et al., 2024](#ref-jaume_hest1k_2024)). Because the cellular and architectural features captured by H&E plausibly reflect the underlying molecular state of the tissue ([He et al., 2020](#ref-he_integrating_2020)), considerable effort has recently gone into predicting gene expression directly from H&E scans ([He et al., 2020](#ref-he_integrating_2020); [Xu et al., 2026](#ref-xu_predicting_2026)). Particularly encouraging are the advances in image-based deep learning and the growing amount of available data ([Nonchev et al., 2026](#ref-nonchev_deepspotm_2026); [Chung et al., 2026](#ref-chung_stpbench_2026)). The road to clinical use is still long, however, because of limited generalisability across cohorts, tissue types and staining protocols, and poor accuracy for lowly expressed genes. [Table 1](#tab-models) summarises a selection of existing models for this task.

<a id="tab-models"></a>

**Table 1. Overview of virtual spatial transcriptomics models that predict gene expression from H&E images.**
*Genes*: predicted gene panel (HVG, highly variable genes; HMHVG, high-mean highly variable genes).
*Encoder → head*: image encoder and the module that maps its features to expression (+, components combined).
*Training data*: as reported by each study, so units differ; a spot is one ST capture area aggregating several cells.
All models predict at spot level except MINT, which predicts single cells.
<sup>\*</sup>MINT additionally takes ST data from the same tissue sample as input and is fitted per sample.
cSCC, cutaneous squamous cell carcinoma; TNBC, triple-negative breast cancer; GNN, graph neural network.

| Model | Genes | Encoder → head | Training data | Tissue |
|:--|:--|:--|:--|:--|
| TRIPLEX<br><sub>([Chung et al., 2024](#ref-chung_accurate_2024))</sub> | 250 top-expressed | ResNet18 → Transformer | ~105k spots | Breast, cSCC |
| DeepSpot-M<br><sub>([Nonchev et al., 2026](#ref-nonchev_deepspotm_2026))</sub> | 19,338 protein-coding | Midnight → Transformer | 730k spots | Pan-cancer (14) |
| Path2Space<br><sub>([Shulman et al., 2024](#ref-shulman_path2space_2024))</sub> | 14,068 detected genes | CTransPath → MLP | ~32k spots | Breast (TNBC) |
| MINT<sup>\*</sup><br><sub>([Chen et al., 2026](#ref-chen_mint_2026))</sub> | Paired-assay genes | UNIv2 + ViT-L → MLP | One section/model | Multi-tissue |
| UNIv2 linear probe<br>(STP-Bench)<br><sub>([Chung et al., 2026](#ref-chung_stpbench_2026))</sub> | HMHVG + markers | UNIv2 → Linear | 609k spots | 6 cancers |
| ST-Net<br><sub>([He et al., 2020](#ref-he_integrating_2020))</sub> | 250 top-expressed | DenseNet-121 (CNN) | 30.6k spots | Breast |
| EGN<br><sub>([Yang et al., 2022](#ref-yang_exemplar_2022))</sub> | 250 top-expressed | StyleGAN + ViT | 30.6k spots | Breast |
| BLEEP<br><sub>([Xie et al., 2023](#ref-xie_spatially_2023))</sub> | 3,467 HVGs (retrieval based) | ResNet50 (CNN) | ~9.3k spots | Healthy liver |
| mclSTExp<br><sub>([Min et al., 2024](#ref-min_multimodal_2024))</sub> | HVGs (retrieval based) | DenseNet-121 + Transformer | 53 sections | Breast, cSCC |
| SEPAL<br><sub>([Mejia et al., 2024](#ref-mejia_sepal_2024))</sub> | 256 spatially variable | ViT-B/16 → GNN | ~20.7k spots | Breast |
| OmiCLIP/Loki<br><sub>([Chen et al., 2025](#ref-chen_visualomics_2025))</sub> | All detected genes (variable) | ViT-L/14 | 2.2M pairs | 32 organs |

Beyond predicting expression, richer molecular readouts open the door to modeling the regulatory logic behind it. Spatially resolved, high-resolution expression data are especially valuable for building generalizable models of cellular regulatory networks, because they capture both which genes a cell expresses and its local tissue context, letting models separate cell-intrinsic regulation from the influence of neighboring cells and signaling gradients ([Dimitrov et al., 2026](#ref-dimitrov_interpretation_2026)). Predicting expression in unseen contexts could then replace broad knock-out screens with a handful of promising candidates, significantly cutting time and cost ([Dimitrov et al., 2026](#ref-dimitrov_interpretation_2026)).

Early perturbation models largely treated cells as independent samples, estimating each cell's response to a stimulus in isolation from its neighbors. Models such as scGen and CPA built on this assumption, learning latent spaces in which perturbation effects could be captured as arithmetic shifts or disentangled from a cell's basal state ([Lotfollahi et al., 2019](#ref-lotfollahi_scgen_2019); [Lotfollahi et al., 2023](#ref-lotfollahi_predicting_2023)). Tissues, however, rarely behave this way: a cell's expression is shaped as much by signals from its neighbors as by its own regulatory program. Later work folded this spatial context back in, using regression- and graph-based approaches such as MISTy and NCEM to quantify neighborhood influence ([Tanevski et al., 2022](#ref-tanevski_explainable_2022); [Fischer et al., 2023](#ref-fischer_modeling_2023)). More recently, a first generation of spatially informed perturbation models, including MintFlow ([Akbarnejad et al., 2025](#ref-akbarnejad_mintflow_2025)), SpatialProp ([Sun et al., 2025](#ref-sun_spatialprop_2025)) and TERRA ([Birk et al., 2026](#ref-birk_terra_2026)), has begun to predict how a cell responds to changes in its microenvironment. However, each supports only one kind of neighborhood intervention (deleting or replacing cells in MintFlow, editing neighbor gene expression in SpatialProp and TERRA), and they are largely evaluated in-domain: MintFlow conditions on cell-type labels and cannot extrapolate to unseen cell types, and SpatialProp is trained separately on each dataset.

In contrast, Cellina ([Moeed et al., 2026](#ref-moeed_querying_2026)), a recently proposed approach, allows much simpler niche perturbations while also supporting reliable out-of-domain applications. Cellina is a graph variational autoencoder that disentangles each cell's gene expression into an intrinsic latent representation, capturing cell identity, and an extrinsic latent representation, encoding the influence of its spatial neighborhood. This allows microenvironment perturbations to be formulated as tractable tissue-graph counterfactuals. Its promising initial results raise a question: can the same disentanglement logic be used to make counterfactual predictions from pathology images alone?

Answering this first of all requires a substitute for Cellina's niche representation. In Cellina, this representation is learned from a graph of the microenvironment in which each node is a cell carrying its gene expression profile as features, and edges connect spatially proximal cells, weighted by physical distance. Without ST this graph cannot be built, so the niche has to be described from the image instead. Recent pathology foundation models are natural candidates for this substitution. Models such as UNIv2, the successor of UNI ([Chen et al., 2024](#ref-chen_towards_2024)), and Virchow2 ([Zimmermann et al., 2024](#ref-zimmermann_virchow2_2024)) build on the DINOv2 ([Oquab et al., 2023](#ref-oquab_dinov2_2023)) self-supervised framework ([Caron et al., 2021](#ref-caron_emerging_2021); [Zhou et al., 2021](#ref-zhou_ibot_2021)), while earlier models such as CTransPath rely on a convolutional–transformer hybrid trained with contrastive learning ([Wang et al., 2022](#ref-wang_transformer_2022)). All of them reduce an image patch to a high-dimensional embedding that, despite being learned without direct molecular supervision, has been shown to be predictive of the underlying molecular state of the tissue ([Chung et al., 2026](#ref-chung_stpbench_2026); [Nonchev et al., 2026](#ref-nonchev_deepspotm_2026)). STP-Bench ([Chung et al., 2026](#ref-chung_stpbench_2026)), a benchmark of 21 virtual ST models re-implemented, where architecturally possible, with a shared image encoder, showed that DeepSpot and TRIPLEX stand out from all other models with significantly better predictions and that the choice of pathology foundation model is a dominant factor in performance, with UNIv2, Virchow2 and H-Optimus-1 ([Bioptimus, 2025](#ref-bioptimus_hoptimus1_2025)) clearly outperforming CTransPath and ImageNet baselines. We therefore focused on the best-performing encoders available in LazySlide ([Zheng et al., 2026](#ref-zheng_lazyslide_2026)), an open-source whole-slide image analysis framework from the scverse ecosystem, which lets all encoders, as well as the Path2Space baseline, run on identical inputs within one pipeline.

The aim of this project was to test whether embeddings from pathology foundation models can stand in for the microenvironment representation that Cellina learns from ST, so that niche counterfactuals could, in principle, be posed from H&E images alone. We worked with a single Xenium Prime section of human ovarian cancer in which every cell has both a measured expression profile and a matched H&E image ([Fig. 1](#fig-overview)), and addressed the question in three steps. First, we benchmarked several pathology foundation models in our own setting: we extracted cell-centred tile embeddings and tested how well a simple linear classifier recovers each cell's tissue domain and cell type from them. This shows which encoder captures the most information about the tissue context. Second, we built a simplified, decoder-only analogue of Cellina that predicts a cell's gene expression from two inputs: its cell-type label, which stands in for the cell's intrinsic state, and a description of its surroundings, given either as a discrete domain label or as the H&E embedding ([Fig. 2](#fig-schematic)). Third, we used this decoder for a counterfactual test. Stromal cells from Smooth muscle were placed *in silico* into a Tumor context by swapping only the context input, and the predicted expression changes were compared with those observed in held-out Tumor regions. Path2Space ([Shulman et al., 2024](#ref-shulman_path2space_2024)), an existing H&E-to-expression ST predictor, served as an external reference. Because the cell-type label is kept as a placeholder for the cell's own state, this setup isolates the question of whether the image can represent the microenvironment; representing the cell itself from H&E is left for future work.

<a id="fig-overview"></a>

<p align="center">
  <img src="docs/figures/fig1_section_overview.png" width="86%" alt="Overview of the ovarian cancer section">
</p>

**Figure 1. Overview of the ovarian cancer section.** Left: H&E-stained FFPE section of human ovarian cancer profiled with 10x Xenium Prime (407,124 segmented cells). Circles mark the 36 regions of interest (ROIs; radius 150 µm), coloured by their annotated tissue domain: Tumor ($`n = 12`$), Smooth muscle ($`n = 12`$), Necrosis ($`n = 2`$), Fallopian tube ($`n = 4`$) and Ovary ($`n = 6`$). The dashed Tumor ROIs 3, 6, 8 and 9 were held out from training. Right: the same section with cells coloured by annotated tissue domain (random 50% of cells shown; unassigned cells omitted). Scale bars, 2 mm.

<br>

---

## 2 Results

### 2.1 Pathology foundation model embeddings encode tissue domain but not cell type

A logistic regression classifier was trained to predict each cell's tissue domain from its tile embedding, for each of the encoders listed in [Table 2a](#tab-domain): Phikon-v2 ([Filiot et al., 2024](#ref-filiot_phikonv2_2024)), UNIv2 ([Chen et al., 2024](#ref-chen_towards_2024)) and Virchow2 ([Zimmermann et al., 2024](#ref-zimmermann_virchow2_2024)). All three models perform similarly well, with UNIv2 and Phikon-v2 slightly ahead of Virchow2. Cell-type prediction was also attempted with the same approach, but with very limited success for all four tested encoders ([Table 2b](#tab-celltype)), suggesting that the embeddings are far more predictive of domain-level features, such as tissue morphology and composition, than of an individual cell's state.

<a id="tab-classification"></a>

**Table 2. Tissue-domain and cell-type classification from pathology foundation model embeddings.** Predictions were made by a multinomial logistic regression classifier on tile embeddings, using the same ROI-based train/test split for both tasks ([Section 3.3](#33-embedding-benchmark)). **a**, Tissue domain (5 classes: Fallopian tube, Necrosis, Ovary, Smooth muscle and Tumor) from Phikon-v2, UNIv2 and Virchow2 embeddings. **b**, Cell type (7 annotated classes) from the same encoders and CTransPath. Balanced accuracy averages the per-class accuracies, so every class counts equally, whereas the weighted F1 and AUROC give more weight to more abundant classes. Best value per column within each panel in bold.

<a id="tab-domain"></a>

**a.** Tissue-domain classification

| Model | Accuracy | Balanced<br>accuracy | F1<br>(macro) | F1<br>(weighted) | AUROC<br>(macro) | AUROC<br>(weighted) |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| Phikon-v2 | **0.973** | 0.967 | 0.962 | **0.973** | **0.999** | **0.998** |
| UNIv2 | 0.969 | **0.970** | **0.973** | 0.969 | **0.999** | **0.998** |
| Virchow2 | 0.898 | 0.938 | 0.856 | 0.911 | 0.997 | 0.994 |

<a id="tab-celltype"></a>

**b.** Cell-type classification

| Model | Accuracy | Balanced<br>accuracy | F1<br>(macro) | F1<br>(weighted) | AUROC<br>(macro) | AUROC<br>(weighted) |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| CTransPath | 0.541 | 0.342 | 0.291 | 0.603 | 0.773 | 0.848 |
| Phikon-v2 | 0.546 | 0.353 | 0.297 | 0.609 | 0.788 | 0.859 |
| UNIv2 | 0.548 | **0.363** | **0.302** | 0.612 | **0.789** | **0.868** |
| Virchow2 | **0.556** | 0.361 | **0.302** | **0.618** | 0.781 | 0.867 |

### 2.2 H&E embeddings support counterfactual prediction of a niche change

To test how well these embeddings substitute for the microenvironment latent variables, we built a mock version of Cellina. Instead of the cell's gene expression, only its cell-type label is passed, and instead of a neighbourhood graph, only its domain label is passed. The two variables are concatenated and decoded into the cell's gene expression; we call this the *labels model*. In the *embeddings model*, the domain label is replaced by the Virchow2 or UNIv2 embedding of the cell's H&E tile, and the same MLP decodes the concatenated cell-type label and embedding into gene expression ([Fig. 2](#fig-schematic)). Both models were then used to predict the expression of Smooth-muscle cells moved into a Tumor context, and the predictions were scored against cells measured in four held-out Tumor ROIs ([Table 3](#tab-counterfactual) and [Fig. 3](#fig-counterfactual)).

For held-out Stromal cells, whose Tumor-context expression was never seen in training (out-of-distribution, OOD), the labels model scores highest on all population-mean metrics (Pearson $`r`$ of logFC 0.725). Among the image-based models, Virchow2 clearly outperforms UNIv2 (0.524 vs. 0.403) and comes close to Path2Space ([Shulman et al., 2024](#ref-shulman_path2space_2024)) (0.636), while achieving the lowest energy distance of all models (9.31 vs. 9.65 for Path2Space). The same ordering holds for the in-distribution (ID) cell types. [Figure 3a](#fig-counterfactual) shows that all models recover the direction of the strongest changes but compress their magnitude, and [Fig. 3b](#fig-counterfactual) shows that the ranking of the models is largely stable across the number $`k`$ of top genes evaluated.

<a id="fig-schematic"></a>

<p align="center">
  <img src="docs/figures/fig2_model_schematic.png" width="100%" alt="Architecture of the embeddings model">
</p>

**Figure 2. Architecture of the embeddings model.** A 224 × 224 px H&E tile (0.5 µm/px, 112 µm) centred on each cell is embedded by a pathology foundation model (UNIv2 or Virchow2) into a $`d`$-dimensional vector, standardised with statistics fitted on training cells only, and concatenated with the one-hot cell-type label ($`\mathbb{R}^7`$) into the conditioning vector $`\mathbf{z}_\text{cond} \in \mathbb{R}^{7+d}`$. The `DecoderSCVI` MLP (two hidden layers of 128 units; Linear → BatchNorm → ReLU) maps it through a softmax to per-gene proportions, which are scaled by the cell's observed library size $`L`$ to give the negative binomial mean $`\mu_g`$. A free, gene-specific dispersion $`\theta_g`$ completes the likelihood, and the model is trained on the negative log-likelihood of the observed counts. The labels model uses the same decoder, with the embedding replaced by the one-hot domain label.

<a id="tab-counterfactual"></a>

**Table 3. Counterfactual prediction of Smooth-muscle cells moved into a Tumor context.** *OOD*: predictions for Stromal cells in the held-out Tumor ROIs; no Stromal cell in a Tumor context was seen during training. *ID*: the same counterfactual for the cell types whose Tumor-context expression was seen during training (all types except Stromal). Path2Space ([Shulman et al., 2024](#ref-shulman_path2space_2024)), a dedicated H&E-to-expression ST predictor, serves as the baseline and predicts the held-out Stromal cells directly from their own H&E tiles; it is a single deterministic run and therefore has no standard deviation. The other models were trained with three random initialisations (mean ± s.d.). Metrics are computed on the 50 genes with the largest observed |logFC| ([Section 3.7](#37-evaluation-metrics)); arrows indicate the better direction. Best value per column within each regime in bold.

| Model | Pearson *r*<br>logFC ↑ | RMSE<br>logFC ↓ | Spearman *ρ*<br>mean expr. ↑ | Precision<br>@50 ↑ | Energy distance<br>(PCA) ↓ |
|:--|:-:|:-:|:-:|:-:|:-:|
| ***Out-of-distribution (held-out Stromal cells)*** | | | | | |
| Labels | **0.725 ± 0.054** | **0.917 ± 0.066** | **0.579 ± 0.059** | **0.447 ± 0.058** | 10.816 ± 0.413 |
| Virchow2 | 0.524 ± 0.045 | 1.166 ± 0.084 | 0.348 ± 0.045 | 0.293 ± 0.023 | **9.308 ± 0.832** |
| UNIv2 | 0.403 ± 0.014 | 1.365 ± 0.028 | 0.213 ± 0.013 | 0.233 ± 0.031 | 12.744 ± 0.262 |
| Path2Space | 0.636 | 1.101 | 0.382 | 0.260 | 9.650 |
| ***In-distribution (other cell types)*** | | | | | |
| Labels | **0.864 ± 0.005** | **1.052 ± 0.006** | **0.834 ± 0.004** | **0.611 ± 0.010** | 11.279 ± 0.027 |
| Virchow2 | 0.677 ± 0.017 | 1.144 ± 0.028 | 0.602 ± 0.024 | 0.520 ± 0.018 | **9.409 ± 0.234** |
| UNIv2 | 0.538 ± 0.012 | 1.073 ± 0.009 | 0.444 ± 0.006 | 0.413 ± 0.012 | 10.379 ± 0.049 |

<a id="fig-counterfactual"></a>

<p align="center">
  <img src="docs/figures/fig3_counterfactual.png" width="100%" alt="Counterfactual prediction of held-out Stromal cells">
</p>

**Figure 3. Counterfactual prediction of held-out Stromal cells moved from Smooth muscle to Tumor.**
**a**, Predicted versus observed expression change. Each panel plots, for every gene, the observed log fold change (logFC) against the logFC predicted by a model (training seed 0 only). Red points are the 50 genes with the largest observed |logFC|. Smooth-muscle Stromal cells are passed through a decoder trained without any Stromal-in-Tumor cell or any cell from the held-out ROIs, with the context switched to Tumor either by the domain label (Labels) or by H&E tile embeddings of Tumor cells (Virchow2, UNIv2). Path2Space ([Shulman et al., 2024](#ref-shulman_path2space_2024)), an external H&E-to-expression ST predictor, predicts Tumor Stromal cells directly from their own H&E tiles and serves as a baseline.
**b**, Benchmark metrics as a function of the number of evaluated genes. Each metric is recomputed on the $`k`$ genes with the largest observed |logFC| for held-out Stromal cells (Smooth muscle → Tumor); the vertical grey line marks $`k = 50`$, the cut-off used in [Table 3](#tab-counterfactual). *Pearson r* and *RMSE* compare observed and predicted logFC. *Spearman* is the rank correlation of mean expression between observed and predicted Tumor Stromal cells. *Precision@k* is the fraction of the $`k`$ genes with the largest predicted |logFC| that are among the $`k`$ largest observed (grey dotted line, random expectation). *Energy distance* compares observed and predicted cell populations in PCA space. Arrows indicate the better direction. Labels, Virchow2 and UNIv2 are counterfactual predictions as in **a**; Path2Space (dashed) predicts directly from each cell's own H&E tile. Lines show the mean over three training seeds; shaded bands, ±1 s.d.

## 3 Methods

### 3.1 Data

We used a publicly available Xenium Prime FFPE human ovarian cancer section from 10x Genomics ([10x Genomics, 2024](#ref-10x_xenium_ovarian_2024)), profiled with the 5K-plex chemistry for the Xenium In Situ platform ([Janesick et al., 2023](#ref-janesick_high_2023)), comprising approximately 407,000 cells and a 5,101-gene panel. The same section was H&E-stained and imaged after Xenium profiling, so each cell has a matched histology image without cross-section alignment. Cell-type labels (malignant, stromal, myeloid, endothelial, lymphoid, epithelial, pericytes) and tissue-domain annotations (Tumor, Smooth muscle, Necrosis, Fallopian tube, Ovary) were taken from the SpatialFusion analysis of the same section ([Yates et al., 2026](#ref-yates_spatialfusion_2026)). For each cell we extracted an H&E tile centered on its centroid ([Section 3.2](#32-histology-processing)) and encoded it with pathology foundation models. The task is to predict the expression of 5,090 genes in each cell from its tile embedding; however, comparison with Path2Space is restricted to the 3,603 genes shared by all models ([Section 3.7](#37-evaluation-metrics)).

We defined 36 circular regions of interest (ROIs; [Fig. 1](#fig-overview)) and restricted the decoder analyses to 24 of them (14,170 cells), holding out four Tumor ROIs for evaluation so that training and test cells never share overlapping H&E context. We chose an ROI-based split because, with a random 80/20 split of cells, the embedding of a held-out cell would be almost identical to that of an overlapping tile from a neighbouring cell seen during training.

### 3.2 Histology processing

All image processing was done with LazySlide (v0.11.1) ([Zheng et al., 2026](#ref-zheng_lazyslide_2026)), an open-source framework for whole-slide image analysis built on the scverse ecosystem. We used it for every step that touches the H&E image, so that all models see identical inputs. The H&E image and cell table were loaded together from a single SpatialData ([Marconato et al., 2025](#ref-marconato_spatialdata_2024)) object (`wsidata.open_wsi`), keeping each cell's tile, expression, cell-type and domain labels linked through one coordinate system. Tiles were defined in physical units rather than pixels: a 224 × 224 px tile at 0.5 µm/px, centred on each cell centroid. This fixes the field of view at 112 × 112 µm regardless of the scan's native resolution. Foundation-model embeddings were extracted with LazySlide's feature-extraction function (`zs.tl.feature_extraction`), which applies each model's own preprocessing. The same tiles were used for the Path2Space baseline ([Section 3.6](#36-path2space-baseline)). Because tiling, preprocessing and feature extraction come from one shared implementation, the models differ only in the encoder and the decoder inputs, not in how the images were prepared.

### 3.3 Embedding benchmark

To compare pathology foundation models in our setting, we tested how well their embeddings separate tissue domains and cell types. Each cell's tile embedding was used as input to a multinomial logistic regression classifier (scikit-learn ([Pedregosa et al., 2011](#ref-pedregosa_scikitlearn_2011)), `lbfgs` solver, `max_iter` = 2000), with features standardised on the training set. We trained two classifiers per encoder: one predicting the tissue domain (5 classes) and one predicting the cell type (7 classes). Cells were split into training and test sets by ROI, so that all cells from one ROI fell on the same side of the split; tiles from neighbouring cells overlap, and a random cell-level split would put the same pixels in both sets. Within each tissue domain, we chose the set of ROIs whose cell count came closest to 20% of that domain's cells and used it as the test set. This gave an approximately 80/20 split for every domain except Necrosis, which had only two ROIs and was therefore split 1:1. Both classifiers used the same ROI split, and cells with no label for a given task were left out of that task. Classes were unevenly represented (malignant cells made up 53% of labelled cells and Tumor 57% of domain-labelled cells, compared with 153 pericytes and 513 Ovary cells), so we report balanced accuracy and macro-averaged F1 and AUROC (one-vs-rest) alongside overall accuracy and weighted scores, and trained with `class_weight="balanced"`.

### 3.4 Decoder models

**Labels model.**
A cell's own expression profile is replaced by its cell-type label, one-hot encoded over the 7 annotated types. Its neighbourhood graph is replaced by its tissue-domain label, one-hot encoded over the 5 domains (Tumor, Smooth muscle, Necrosis, Fallopian tube, Ovary). The two one-hot vectors are concatenated into a 12-dimensional conditioning vector

```math
\mathbf{z}_n = \big[\, \mathbf{e}(\text{cell type}_n) \,\|\, \mathbf{d}(\text{domain}_n) \,\big]. \tag{1}
```

**Embeddings model.**
This is the same model with the domain one-hot replaced by a continuous image representation of the cell's surroundings. The H&E tile around each cell ([Section 3.2](#32-histology-processing)) is embedded with a pretrained pathology foundation model: Virchow2 (2,560-d) or UNIv2 (1,536-d). Each embedding feature is standardised using statistics from the training cells only. The embedding is concatenated with the cell-type one-hot, giving a conditioning vector of 2,567 (Virchow2) or 1,543 (UNIv2) dimensions ([Fig. 2](#fig-schematic)).

**Decoder.**
Both models share the same decoder, a two-layer MLP (scvi-tools `DecoderSCVI`; [Lopez et al., 2018](#ref-lopez_deep_2018); [Gayoso et al., 2022](#ref-gayoso_python_2022)). Each hidden layer has 128 units with batch normalisation and ReLU activation, and no dropout, so the conditioning vector is compressed into a 128-dimensional hidden representation. A linear layer followed by a softmax maps it to a proportion vector $`\boldsymbol{\rho}_n`$ over the 5,090 panel genes. Expected counts are the proportions scaled by the cell's observed library size $`\ell_n`$ (total counts), $`\mu_{ng} = \ell_n \rho_{ng}`$. Counts are modelled with a negative binomial distribution,

```math
x_{ng} \sim \mathrm{NB}(\mu_{ng}, \theta_g), \tag{2}
```

where $`\theta_g`$ is a learned gene-specific inverse dispersion. Neither model has an encoder or a latent variable. Both are trained by minimising the negative binomial negative log-likelihood of the observed counts, summed over genes and averaged over the $`N`$ cells in a minibatch,

```math
\mathcal{L} = -\frac{1}{N} \sum_{n=1}^{N} \sum_{g} \log \mathrm{NB}\!\left(x_{ng} \mid \mu_{ng}, \theta_g\right), \tag{3}
```

using Adam ([Kingma and Ba, 2014](#ref-kingma_adam_2014)) (learning rate $`10^{-3}`$, minibatches of 2,048 cells, up to 100 epochs, early stopping on validation loss with patience 10). Each model was trained with three random seeds.

### 3.5 Counterfactual prediction

To predict how Smooth-muscle cells would appear in a Tumor environment, we kept each cell's own cell-type label and library size and changed only the context input. In the labels model, the domain one-hot is set to Tumor. The embeddings model has no discrete "Tumor" input, so each cell's embedding is replaced with that of a randomly drawn Tumor cell (the donor). The held-out cell type is excluded from the donors, and only the donor's embedding is used, never its expression. We drew donors from all Tumor cells rather than only from Stromal cells because the held-out ROIs contain only around 500 Stromal cells; the larger donor pool allows many more distinct counterfactual draws.

### 3.6 Path2Space baseline

As a baseline for our predictions we chose the Path2Space ST predictor ([Shulman et al., 2024](#ref-shulman_path2space_2024)). Although in the STP-Bench benchmark ([Chung et al., 2026](#ref-chung_stpbench_2026)) it ranks behind models such as TRIPLEX ([Chung et al., 2024](#ref-chung_accurate_2024)) and DeepSpot ([Nonchev et al., 2025](#ref-nonchev_deepspot_2025)), and only marginally above the UNIv2 linear-probe baseline, it was the most stable workflow to implement within LazySlide at the time. Path2Space was trained only on CTransPath ([Wang et al., 2022](#ref-wang_transformer_2022)) embeddings, so we provided CTransPath embeddings of the same tiles to the pretrained Path2Space decoder.

### 3.7 Evaluation metrics

Predictions were compared with measured expression at the level of cell populations. Before comparison, predictions were restricted to the genes shared by Path2Space and the other models, resulting in 3,603 evaluated genes, although all four models are able to predict a wider gene set. The observed data are raw Xenium counts $`x_{ng}`$ for cell $`n`$ and gene $`g`$, split into two sets:

- the control population $`\mathcal{C}`$: cells in the source domain (Smooth muscle);
- the target population $`\mathcal{T}`$: cells in the target domain (Tumor) within the held-out ROIs.

The predicted population $`\mathcal{P}`$ consists of one predicted expression profile per cell. For the decoder-based models this is the mean of the negative binomial, $`\mu_{ng} = \ell_n \rho_{ng}`$, where $`\ell_n`$ is the cell's observed library size and $`\rho_{ng}`$ is the decoder output. For Path2Space, predictions $`\hat{y}`$ on its native $`\log_{10}(x+1)`$ scale were converted back to pseudo-counts as $`10^{\hat{y}} - 1`$.

Observed and predicted profiles were normalised identically. Each cell was scaled to $`10^4`$ total counts over the evaluated gene set and then log-transformed:

```math
y_{ng} = \log\!\left(1 + 10^{4}\,\frac{x_{ng}}{\sum_{g'} x_{ng'}}\right). \tag{4}
```

For any population $`\mathcal{S}`$, the mean log-normalised expression of gene $`g`$ is $`\bar{y}^{\mathcal{S}}_g = \frac{1}{|\mathcal{S}|}\sum_{n\in\mathcal{S}} y_{ng}`$.

**Log fold change.**
The observed and predicted log fold changes (logFC) of gene $`g`$ were both defined relative to the *observed* control population:

```math
\Delta^{\mathrm{obs}}_g = \bar{y}^{\mathcal{T}}_g - \bar{y}^{\mathcal{C}}_g,
\qquad
\Delta^{\mathrm{pred}}_g = \bar{y}^{\mathcal{P}}_g - \bar{y}^{\mathcal{C}}_g . \tag{5}
```

Using the same reference for both means that differences between $`\Delta^{\mathrm{obs}}`$ and $`\Delta^{\mathrm{pred}}`$ reflect only the predicted target population, not the model's reconstruction of the control population.

**Evaluated genes.**
Most metrics were computed on the set $`\mathcal{D}`$ of the $`K = 50`$ genes with the largest observed absolute logFC, $`|\Delta^{\mathrm{obs}}_g|`$. This set was defined from observed data only and was therefore identical for all models within an evaluation.

**Pearson correlation of logFC.**
This metric measures whether the model reproduces the direction and relative size of the expression changes:

```math
r_{\mathrm{logFC}} = \operatorname{corr}_{g\in\mathcal{D}}\!\left(\Delta^{\mathrm{obs}}_g,\; \Delta^{\mathrm{pred}}_g\right). \tag{6}
```

**Root mean squared error of logFC.**
This metric measures the absolute agreement of the effect sizes, so it also penalises systematic offsets and scaling errors that the correlation does not detect:

```math
\mathrm{RMSE}_{\mathrm{logFC}} = \sqrt{\frac{1}{K}\sum_{g\in\mathcal{D}}\left(\Delta^{\mathrm{obs}}_g - \Delta^{\mathrm{pred}}_g\right)^2 }. \tag{7}
```

**Spearman correlation of mean expression.**
This metric measures whether the model correctly ranks the absolute expression levels of the evaluated genes in the target population. It uses the population means themselves, with no control subtracted:

```math
\rho_{\mathrm{avg}} = \operatorname{corr}^{\mathrm{Spearman}}_{g\in\mathcal{D}}\!\left(\bar{y}^{\mathcal{T}}_g,\; \bar{y}^{\mathcal{P}}_g\right). \tag{8}
```

**Precision at top 50.**
This metric measures whether the model identifies the most strongly changing genes. Let $`\hat{\mathcal{D}}`$ be the $`K`$ genes with the largest predicted absolute logFC, $`|\Delta^{\mathrm{pred}}_g|`$, selected from the full evaluated gene set rather than from $`\mathcal{D}`$. Precision is the fraction of shared genes:

```math
\mathrm{P@}K = \frac{|\mathcal{D} \cap \hat{\mathcal{D}}|}{K}. \tag{9}
```

Because both sets contain $`K`$ genes, this value also equals recall. Both the sign of the change and the order within the top-$`K`$ sets are ignored. A random ranking of $`G`$ genes gives an expected value of $`K/G`$.

**Energy distance.**
Unlike the metrics above, which compare population averages, the energy distance ([Székely and Rizzo, 2013](#ref-szekely_energy_2013)) compares the full distributions of observed and predicted cells. Each cell was represented by its log-normalised expression of the genes in $`\mathcal{D}`$, projected onto the principal components fitted on the observed target cells. With 50 genes and 50 components, this projection is an orthogonal rotation and leaves distances unchanged. For observed cells $`\{\mathbf{a}_i\}_{i=1}^{m}`$ and predicted cells $`\{\mathbf{b}_j\}_{j=1}^{n}`$,

```math
E(\mathcal{T},\mathcal{P}) = \frac{2}{mn}\sum_{i=1}^{m}\sum_{j=1}^{n}\lVert \mathbf{a}_i-\mathbf{b}_j\rVert_2
- \frac{1}{m^2}\sum_{i=1}^{m}\sum_{i'=1}^{m}\lVert \mathbf{a}_i-\mathbf{a}_{i'}\rVert_2
- \frac{1}{n^2}\sum_{j=1}^{n}\sum_{j'=1}^{n}\lVert \mathbf{b}_j-\mathbf{b}_{j'}\rVert_2 . \tag{10}
```

$`E \ge 0`$, and in the population limit $`E = 0`$ if and only if the two distributions are identical. To limit computational cost, each population was randomly subsampled to at most 2,000 cells. This was repeated 10 times with a fixed random seed, and the mean is reported. The predicted profiles are expected values rather than sampled counts, so they lack the technical sampling noise of measured cells. The energy distance therefore also penalises this reduced cell-to-cell variability, in addition to differences in location.

**Aggregation.**
Each trained model was evaluated under three random seeds (0, 1, 2), which differed in network initialisation and, for the embedding-conditioned counterfactuals, in the draw of donor embeddings. We report the mean ± standard deviation across seeds. Path2Space was used pretrained and is deterministic, so it was evaluated once.

<br>

---

## 4 Discussion

Despite its simplicity, a linear classifier recovered the tissue domain from pathology foundation model embeddings with almost perfect accuracy, showing that microenvironment information can be reliably inferred from these embeddings and therefore used as a substitute for a microenvironment latent. Interestingly, although UNIv2 performed slightly better than Virchow2 in domain classification, its embeddings were weaker than Virchow2's for predicting gene expression. With only the limited training data that a single slide provides, the Virchow2 embeddings model approaches the performance of the Path2Space ST predictor ([Shulman et al., 2024](#ref-shulman_path2space_2024)) and matches it on energy distance.

Because most of our metrics compare population-level averages of logFC, it is not surprising that the labels model performs best: it only learns the most likely expression profile of each cell-type–domain pair, so it is effectively built to win on population-mean metrics. It therefore serves as a basic proof-of-concept model rather than a true baseline. The true baseline was Path2Space; stronger ST predictors exist, such as TRIPLEX ([Chung et al., 2024](#ref-chung_accurate_2024)), which ranks above Path2Space in STP-Bench ([Chung et al., 2026](#ref-chung_stpbench_2026)), and DeepSpot-M ([Nonchev et al., 2026](#ref-nonchev_deepspotm_2026)), which outperforms its predecessor DeepSpot. Comparing our performance against them will be essential in future work.

Although this was a small-scale test with limited success, it showed that embeddings from pathology foundation models can be used to predict counterfactual gene expression nearly as well as an established ST predictor. These results are encouraging, particularly given the small sample size and the very simple MLP decoder, which nonetheless comes close to the performance of a dedicated H&E-to-expression model. To make this approach truly work, however, a suitable substitute for the cell latent must be found, which will be the next step of this research, carried out as part of a Master's thesis. There are obvious features one could use to describe a cell, such as those derived from its shape or staining. There are also tools that can be leveraged, such as CytoFormer ([Yao et al., 2026](#ref-yao_cytoformer_2026)), which is designed to classify cell types directly from H&E. Once both the cell and its microenvironment can be reliably represented from H&E alone, what remains is to find the architecture best suited to decoupling cell-intrinsic from extrinsic influences, and the best data on which to train it.

<br>

---

## References

<a id="ref-10x_xenium_ovarian_2024"></a>10x Genomics (2024). *Xenium Prime FFPE Human Ovarian Cancer*. Dataset. url: <https://www.10xgenomics.com/datasets/xenium-prime-ffpe-human-ovarian-cancer>. (visited on 2026-10-09).

<a id="ref-akbarnejad_mintflow_2025"></a>Akbarnejad, A., Steele, L., Jafree, D. J. et al. (2025). “Mapping and reprogramming human tissue microenvironments with MintFlow”. bioRxiv preprint. doi: [10.1101/2025.06.24.661094](https://doi.org/10.1101/2025.06.24.661094).

<a id="ref-bioptimus_hoptimus1_2025"></a>Bioptimus (2025). *H-optimus-1*. Model card. url: <https://huggingface.co/bioptimus/H-optimus-1>. (visited on 2026-10-09).

<a id="ref-birk_terra_2026"></a>Birk, S., Sanian, M. V., Vahidi, A. et al. (2026). “Multi-scale modeling of human tissues from spatial transcriptomics with TERRA”. bioRxiv preprint. doi: [10.64898/2026.07.29.741565](https://doi.org/10.64898/2026.07.29.741565).

<a id="ref-bressan_dawn_2023"></a>Bressan, D., Battistoni, G. and Hannon, G. J. (2023). “The dawn of spatial omics”. *Science* 381(6657), eabq4964. doi: [10.1126/science.abq4964](https://doi.org/10.1126/science.abq4964).

<a id="ref-caron_emerging_2021"></a>Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P. and Joulin, A. (2021). “Emerging Properties in Self-Supervised Vision Transformers”. arXiv preprint. doi: [10.48550/arXiv.2104.14294](https://doi.org/10.48550/arXiv.2104.14294).

<a id="ref-chen_mint_2026"></a>Chen, L., Zhang, Z., Wang, B. et al. (2026). “MINT infers the latent single-cell spatial transcriptome from paired histology”. bioRxiv preprint. doi: [10.64898/2026.09.16.751717](https://doi.org/10.64898/2026.09.16.751717).

<a id="ref-chen_towards_2024"></a>Chen, R. J., Ding, T., Lu, M. Y. et al. (2024). “Towards a general-purpose foundation model for computational pathology”. *Nature Medicine* 30(3), pp. 850–862. doi: [10.1038/s41591-024-02857-3](https://doi.org/10.1038/s41591-024-02857-3).

<a id="ref-chen_visualomics_2025"></a>Chen, W., Zhang, P., Tran, T. N. et al. (2025). “A visual–omics foundation model to bridge histopathology with spatial transcriptomics”. *Nature Methods* 22(7), pp. 1568–1582. doi: [10.1038/s41592-025-02707-1](https://doi.org/10.1038/s41592-025-02707-1).

<a id="ref-chung_accurate_2024"></a>Chung, Y., Ha, J. H., Im, K. C. and Lee, J. S. (2024). “Accurate Spatial Gene Expression Prediction by integrating Multi-resolution features”. In: *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 11591–11600. doi: [10.48550/arXiv.2403.07592](https://doi.org/10.48550/arXiv.2403.07592).

<a id="ref-chung_stpbench_2026"></a>Chung, Y., Ha, J. H., Song, A. H. et al. (2026). “STP-BENCH: A Unified Systematic Benchmark for Virtual Spatial Transcriptomics from Histopathology Images”. arXiv preprint. doi: [10.48550/arXiv.2609.05956](https://doi.org/10.48550/arXiv.2609.05956).

<a id="ref-dimitrov_interpretation_2026"></a>Dimitrov, D., Schrod, S., Rohbeck, M. and Stegle, O. (2026). “Interpretation, extrapolation and perturbation of single cells”. *Nature Reviews Genetics* 27(5), pp. 349–370. doi: [10.1038/s41576-025-00920-4](https://doi.org/10.1038/s41576-025-00920-4).

<a id="ref-filiot_phikonv2_2024"></a>Filiot, A., Jacob, P., Kain, A. M. and Saillard, C. (2024). “Phikon-v2, A large and public feature extractor for biomarker prediction”. arXiv preprint. doi: [10.48550/arXiv.2409.09173](https://doi.org/10.48550/arXiv.2409.09173).

<a id="ref-fischer_modeling_2023"></a>Fischer, D. S., Schaar, A. C. and Theis, F. J. (2023). “Modeling intercellular communication in tissues using spatial graphs of cells”. *Nature Biotechnology* 41(3), pp. 332–336. doi: [10.1038/s41587-022-01467-z](https://doi.org/10.1038/s41587-022-01467-z).

<a id="ref-gayoso_python_2022"></a>Gayoso, A., Lopez, R., Xing, G. et al. (2022). “A Python library for probabilistic analysis of single-cell omics data”. *Nature Biotechnology* 40(2), pp. 163–166. doi: [10.1038/s41587-021-01206-w](https://doi.org/10.1038/s41587-021-01206-w).

<a id="ref-he_integrating_2020"></a>He, B., Bergenstråhle, L., Stenbeck, L., Abid, A., Andersson, A., Borg, Å., Maaskola, J., Lundeberg, J. and Zou, J. (2020). “Integrating spatial gene expression and breast tumour morphology via deep learning”. *Nature Biomedical Engineering* 4(8), pp. 827–834. doi: [10.1038/s41551-020-0578-x](https://doi.org/10.1038/s41551-020-0578-x).

<a id="ref-janesick_high_2023"></a>Janesick, A., Shelansky, R., Gottscho, A. D. et al. (2023). “High resolution mapping of the tumor microenvironment using integrated single-cell, spatial and in situ analysis”. *Nature Communications* 14(1), 8353. doi: [10.1038/s41467-023-43458-x](https://doi.org/10.1038/s41467-023-43458-x).

<a id="ref-jaume_hest1k_2024"></a>Jaume, G., Doucet, P., Song, A. H. et al. (2024). “HEST-1k: A Dataset for Spatial Transcriptomics and Histology Image Analysis”. arXiv preprint. doi: [10.48550/arXiv.2406.16192](https://doi.org/10.48550/arXiv.2406.16192).

<a id="ref-kingma_adam_2014"></a>Kingma, D. P. and Ba, J. (2014). “Adam: A Method for Stochastic Optimization”. arXiv preprint. doi: [10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980).

<a id="ref-lewis_spatial_2021"></a>Lewis, S. M., Asselin-Labat, M.-L., Nguyen, Q., Berthelet, J., Tan, X., Wimmer, V. C., Merino, D., Rogers, K. L. and Naik, S. H. (2021). “Spatial omics and multiplexed imaging to explore cancer biology”. *Nature Methods* 18(9), pp. 997–1012. doi: [10.1038/s41592-021-01203-6](https://doi.org/10.1038/s41592-021-01203-6).

<a id="ref-lopez_deep_2018"></a>Lopez, R., Regier, J., Cole, M. B., Jordan, M. I. and Yosef, N. (2018). “Deep generative modeling for single-cell transcriptomics”. *Nature Methods* 15(12), pp. 1053–1058. doi: [10.1038/s41592-018-0229-2](https://doi.org/10.1038/s41592-018-0229-2).

<a id="ref-lotfollahi_predicting_2023"></a>Lotfollahi, M., Klimovskaia Susmelj, A., De Donno, C. et al. (2023). “Predicting cellular responses to complex perturbations in high-throughput screens”. *Molecular Systems Biology* 19(6), MSB202211517. doi: [10.15252/msb.202211517](https://doi.org/10.15252/msb.202211517).

<a id="ref-lotfollahi_scgen_2019"></a>Lotfollahi, M., Wolf, F. A. and Theis, F. J. (2019). “scGen predicts single-cell perturbation responses”. *Nature Methods* 16(8), pp. 715–721. doi: [10.1038/s41592-019-0494-8](https://doi.org/10.1038/s41592-019-0494-8).

<a id="ref-marconato_spatialdata_2024"></a>Marconato, L., Palla, G., Yamauchi, K. A. et al. (2025). “SpatialData: an open and universal data framework for spatial omics”. *Nature Methods* 22(1), pp. 58–62. doi: [10.1038/s41592-024-02212-x](https://doi.org/10.1038/s41592-024-02212-x).

<a id="ref-mejia_sepal_2024"></a>Mejia, G., Cárdenas, P., Ruiz, D., Castillo, A. and Arbeláez, P. (2024). “SEPAL: Spatial Gene Expression Prediction from Local Graphs”. arXiv preprint. doi: [10.48550/arXiv.2309.01036](https://doi.org/10.48550/arXiv.2309.01036).

<a id="ref-min_multimodal_2024"></a>Min, W., Shi, Z., Zhang, J., Wan, J. and Wang, C. (2024). “Multimodal contrastive learning for spatial gene expression prediction using histology images”. arXiv preprint. doi: [10.48550/arXiv.2407.08216](https://doi.org/10.48550/arXiv.2407.08216).

<a id="ref-moeed_querying_2026"></a>Moeed, A., Schrod, S., Rohbeck, M., Bonder, M. J., Lutsik, P., Stegle, O. and Dimitrov, D. (2026). “Querying Counterfactuals on Tissue Graphs with Supervised Disentanglement”. arXiv preprint. doi: [10.48550/arXiv.2606.08493](https://doi.org/10.48550/arXiv.2606.08493).

<a id="ref-nonchev_deepspotm_2026"></a>Nonchev, K., Dawo, S., Silina, K., Koelzer, V. H. and Rätsch, G. (2026). “DeepSpot-M: a multimodal foundation model for transcriptome-wide virtual spatial transcriptomics from histology”. medRxiv preprint. doi: [10.64898/2026.06.19.26356060](https://doi.org/10.64898/2026.06.19.26356060).

<a id="ref-nonchev_deepspot_2025"></a>Nonchev, K., Dawo, S., Silina, K., Moch, H., Andani, S., Tumor Profiler Consortium, Koelzer, V. H. and Rätsch, G. (2025). “DeepSpot: Leveraging Spatial Context for Enhanced Spatial Transcriptomics Prediction from H&E Images”. medRxiv preprint. doi: [10.1101/2025.02.09.25321567](https://doi.org/10.1101/2025.02.09.25321567).

<a id="ref-oquab_dinov2_2023"></a>Oquab, M., Darcet, T., Moutakanni, T. et al. (2023). “DINOv2: Learning Robust Visual Features without Supervision”. arXiv preprint. doi: [10.48550/arXiv.2304.07193](https://doi.org/10.48550/arXiv.2304.07193).

<a id="ref-pedregosa_scikitlearn_2011"></a>Pedregosa, F., Varoquaux, G., Gramfort, A. et al. (2011). “Scikit-learn: Machine Learning in Python”. *Journal of Machine Learning Research* 12, pp. 2825–2830. url: <https://jmlr.org/papers/v12/pedregosa11a.html>.

<a id="ref-shulman_path2space_2024"></a>Shulman, E. D., Campagnolo, E. M., Lodha, R., Cantore, T., Hu, T., Nasrallah, M., Hoang, D.-T., Aldape, K. and Ruppin, E. (2024). “Path2Space: An AI Approach for Cancer Biomarker Discovery Via Histopathology Inferred Spatial Transcriptomics”. bioRxiv preprint, version 1. doi: [10.1101/2024.10.16.618609](https://doi.org/10.1101/2024.10.16.618609).

<a id="ref-sun_spatialprop_2025"></a>Sun, E. D., Buendia, A., Brunet, A. and Zou, J. (2025). “SpatialProp: tissue perturbation modeling with spatially resolved single-cell transcriptomics”. bioRxiv preprint. doi: [10.64898/2025.11.30.691355](https://doi.org/10.64898/2025.11.30.691355).

<a id="ref-szekely_energy_2013"></a>Székely, G. J. and Rizzo, M. L. (2013). “Energy statistics: A class of statistics based on distances”. *Journal of Statistical Planning and Inference* 143(8), pp. 1249–1272. doi: [10.1016/j.jspi.2013.03.018](https://doi.org/10.1016/j.jspi.2013.03.018).

<a id="ref-tanevski_explainable_2022"></a>Tanevski, J., Flores, R. O. R., Gabor, A., Schapiro, D. and Saez-Rodriguez, J. (2022). “Explainable multiview framework for dissecting spatial relationships from highly multiplexed data”. *Genome Biology* 23(1), 97. doi: [10.1186/s13059-022-02663-5](https://doi.org/10.1186/s13059-022-02663-5).

<a id="ref-wang_transformer_2022"></a>Wang, X., Yang, S., Zhang, J., Wang, M., Zhang, J., Yang, W., Huang, J. and Han, X. (2022). “Transformer-based unsupervised contrastive learning for histopathological image classification”. *Medical Image Analysis* 81, p. 102559. doi: [10.1016/j.media.2022.102559](https://doi.org/10.1016/j.media.2022.102559).

<a id="ref-xie_spatially_2023"></a>Xie, R., Pang, K., Chung, S. W., Perciani, C. T., MacParland, S. A., Wang, B. and Bader, G. D. (2023). “Spatially Resolved Gene Expression Prediction from H&E Histology Images via Bi-modal Contrastive Learning”. arXiv preprint. doi: [10.48550/arXiv.2306.01859](https://doi.org/10.48550/arXiv.2306.01859).

<a id="ref-xu_predicting_2026"></a>Xu, K., Koohy, H. and Antanaviciute, A. (2026). “Predicting spatial transcriptomics from histology images: Progress, challenges, and directions”. *EXO*. doi: [10.70401/EXO.2026.0020](https://doi.org/10.70401/EXO.2026.0020).

<a id="ref-yang_exemplar_2022"></a>Yang, Y., Hossain, M. Z., Stone, E. A. and Rahman, S. (2022). “Exemplar Guided Deep Neural Network for Spatial Transcriptomics Analysis of Gene Expression Prediction”. arXiv preprint. doi: [10.48550/arXiv.2210.16721](https://doi.org/10.48550/arXiv.2210.16721).

<a id="ref-yao_cytoformer_2026"></a>Yao, J., Li, S., Yu, A. and Huang, Z. (2026). “CytoFormer: A Molecularly Supervised Cell Foundation Model for Histopathology Cell Classification”. arXiv preprint. doi: [10.48550/arXiv.2608.16718](https://doi.org/10.48550/arXiv.2608.16718).

<a id="ref-yates_spatialfusion_2026"></a>Yates, J., Shavakhi, M., Choueiri, T. K., Van Allen, E. M. and Uhler, C. (2026). “SpatialFusion: A lightweight multimodal foundation model for pathway-informed spatial niche mapping”. bioRxiv preprint. doi: [10.64898/2026.03.16.712056](https://doi.org/10.64898/2026.03.16.712056).

<a id="ref-zheng_lazyslide_2026"></a>Zheng, Y., Abila, E., Chrenková, E., Buljan, I., Winkler, J. and Rendeiro, A. F. (2026). “LazySlide: accessible and interoperable whole-slide image analysis”. *Nature Methods* 23(4), pp. 728–731. doi: [10.1038/s41592-026-03044-7](https://doi.org/10.1038/s41592-026-03044-7).

<a id="ref-zhou_ibot_2021"></a>Zhou, J., Wei, C., Wang, H., Shen, W., Xie, C., Yuille, A. and Kong, T. (2021). “iBOT: Image BERT Pre-Training with Online Tokenizer”. arXiv preprint. doi: [10.48550/arXiv.2111.07832](https://doi.org/10.48550/arXiv.2111.07832).

<a id="ref-zimmermann_virchow2_2024"></a>Zimmermann, E., Vorontsov, E., Viret, J. et al. (2024). “Virchow2: Scaling Self-Supervised Mixed Magnification Models in Pathology”. arXiv preprint. doi: [10.48550/arXiv.2408.00738](https://doi.org/10.48550/arXiv.2408.00738).

## Declaration on the use of AI tools

Claude AI, developed by Anthropic, was used as an auxiliary tool to provide guidance during the coding process, generate code independently for specific tasks, and assist in creating and refining plots for data visualization. It was also used to rephrase selected sections of the written report to improve clarity and readability. Responsibility for the final content, code, methodology, and academic quality of the submitted work remains with the author.
