# Crohn’s Disease Microbiome Analysis using 16S rRNA Data

This project explores gut microbiome differences between Crohn’s disease (CD) patients and healthy controls using 16S rRNA gene sequencing. Fecal samples from 60 individuals (30 CD, 30 controls), all of Ashkenazi Jewish descent from the New York area, provided by UCLA via NCBI's sequence read archive (SRA), were analyzed using QIIME2 to investigate microbial diversity, taxonomic composition, and differentially abundant taxa associated with Crohn’s disease.



##  Objectives

- Investigate microbiome composition in Crohn’s disease
- Compare alpha diversity (Faith’s PD, Shannon index)
- Assess beta diversity using UniFrac distances and PCoA
- Identify differentially abundant taxa between groups

---

## Key Findings

- **Reduced alpha diversity** in CD patients (p < 0.01), suggesting microbial dysbiosis.
- **Distinct beta diversity clustering** observed via UniFrac PCoA, confirmed by PERMANOVA (pseudo-F = 5.05, p = 0.001).
- **Differentially abundant genera** identified, supporting potential for microbiome-based diagnostics.

---

## Live Dashboard

Explore interactive Streamlit dashboard here: [![View Dashboard](https://img.shields.io/badge/Live%20Dashboard-Streamlit-green?logo=streamlit)](https://tadiwos-crohns-metagenomics.streamlit.app/)

## Tools & Technologies

- **QIIME2 2024.5** for microbiome analysis  
- **Python 3** with **Jupyter Notebook** for data exploration and reporting  
- **16S rRNA gene sequencing** (Illumina) — data sourced from public repositories (SRA) using the SRA Toolkit  

###  Environment Setup
- Cross-platform users can use the provided `qiime2_env.yml` to recreate the analysis environment with Conda  
- **Windows users are recommended to use [WSL (Windows Subsystem for Linux)](https://learn.microsoft.com/en-us/windows/wsl/install)** in combination with `qiime2_env.yml` for a smooth QIIME2 setup.
- Alternative: use **Docker** with the latest QIIME2 image for containerized execution
  
---

##  Setup Instructions

To recreate the qiime2 environment:

```bash
conda env create -f environment/qiime2_env.yml
conda activate qiime2_env

```

###  Download Required File

Due to size restrictions, the `demux-single-end.qza` file (~600MB) is not included in this repository.

You can download it from the link below and place it in the `data/` directory:

[Download demux-single-end.qza](https://drive.google.com/file/d/1D6tbxXNTYO7lXK9P7TaEBszO-73g_dcm/view?usp=sharing)

> Make sure the filename remains `demux-single-end.qza` and is located in the `data/` folder to ensure the notebook runs smoothly.


## Conclusion

This project underscores the role of gut microbiome shifts in Crohn’s disease, revealing reduced microbial diversity and distinct taxonomic differences in CD patients. These findings align with current literature and support the growing interest in microbiome-based diagnostics and therapies for IBD. Further studies are needed to explore the functional impact of these microbial changes on disease progression.

## Future Directions

Future research could explore the functional consequences of the microbiome shifts observed in Crohn’s disease and their roles in disease progression. Shotgun metagenomic sequencing may offer deeper insight by capturing strain-level variation and functional potential, supporting the discovery of more precise microbial biomarkers and therapeutic targets.

## Contributions

Contributions are welcome!

You’re encouraged to:
- Extend the microbiome analysis (e.g., functional pathways, strain-level insights)
- Suggest improvements to visualizations or data presentation
- Report bugs or request new features
- Start discussions on potential extensions or workflow enhancements

Feel free to open an issue or submit a pull request with your ideas or improvements.

---

### Acknowledgements

- The data used in this project was sourced from the **Sequence Read Archive (SRA)** and was generously provided by **UCLA**.
- Special thanks to the **QIIME2** development team for their powerful toolset used in the analysis.

