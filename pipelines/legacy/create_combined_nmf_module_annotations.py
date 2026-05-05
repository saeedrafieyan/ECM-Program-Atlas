from __future__ import annotations

from pathlib import Path
import pandas as pd


BASE_PATH = Path(
    "outputs/latent_baseline_embeddings/"
    "rna_tissue_consensus/"
    "combined_nmf_module_summary_core_glycoproteins_proteoglycans_collagens.csv"
)

OUTPUT_PATH = Path(
    "outputs/latent_baseline_embeddings/"
    "rna_tissue_consensus/"
    "combined_nmf_module_annotations.csv"
)


ANNOTATIONS = {
    # Core Matrisome
    ("core_matrisome", "NMF1"): (
        "Vascular-stromal connective ECM",
        "High",
        "Vascular, stromal, and connective ECM program enriched in tongue, heart muscle, skeletal muscle, adipose tissue, blood vessel, and breast."
    ),
    ("core_matrisome", "NMF2"): (
        "Retinal-neural specialized ECM",
        "Moderate-high",
        "Retinal and neural-specialized ECM module enriched in retina and supported by genes such as COL25A1, NYX, OPTC, NELL2, and COL4A3."
    ),
    ("core_matrisome", "NMF3"): (
        "Stromal remodeling and basement membrane ECM",
        "High",
        "Stromal remodeling and basement membrane-associated ECM enriched in placenta, lung, rectum, gallbladder, smooth muscle, and urinary bladder."
    ),
    ("core_matrisome", "NMF4"): (
        "Reproductive-specialized basement membrane ECM",
        "Moderate",
        "Specialized reproductive and epithelial basement membrane module enriched in epididymis, seminal vesicle, prostate, fallopian tube, and endometrium."
    ),
    ("core_matrisome", "NMF5"): (
        "Epithelial-mucosal basement membrane ECM",
        "High",
        "Epithelial and mucosal basement membrane module enriched in skin, vagina, salivary gland, stomach, esophagus, cervix, colon, and small intestine."
    ),
    ("core_matrisome", "NMF6"): (
        "CNS neural ECM",
        "Very high",
        "Strong CNS ECM module enriched in cerebral cortex, amygdala, hippocampal formation, midbrain, basal ganglia, hypothalamus, spinal cord, and cerebellum."
    ),
    ("core_matrisome", "NMF7"): (
        "Immune-lymphoid remodeling ECM",
        "High",
        "Immune, lymphoid, and mucosal remodeling ECM module enriched in appendix, tonsil, lymph node, thymus, rectum, small intestine, and spleen."
    ),
    ("core_matrisome", "NMF8"): (
        "Renal-endothelial basement membrane ECM",
        "High",
        "Renal, endothelial, and basement membrane module enriched in kidney, pituitary gland, thyroid gland, lung, pancreas, prostate, and blood vessel."
    ),
    ("core_matrisome", "NMF9"): (
        "Hepatic plasma-associated ECM",
        "High",
        "Hepatic and plasma-protein-associated ECM module enriched in liver, pancreas, spleen, adrenal gland, and adipose tissue."
    ),
    ("core_matrisome", "NMF10"): (
        "Fibroblastic collagen-rich interstitial ECM",
        "High",
        "Fibroblastic and collagen-rich interstitial ECM module enriched in ovary, blood vessel, gallbladder, endometrium, smooth muscle, cervix, and adipose tissue."
    ),

    # ECM Glycoproteins
    ("ecm_glycoproteins", "NMF1"): (
        "GI-mucosal glycoprotein ECM",
        "High",
        "Glycoprotein module enriched in small intestine, rectum, appendix, duodenum, colon, gallbladder, stomach, and urinary bladder."
    ),
    ("ecm_glycoproteins", "NMF2"): (
        "Retinal-neural glycoprotein ECM",
        "Moderate-high",
        "Retinal, renal, and neural-associated glycoprotein module with genes such as NTNG1, ZPLD1, MATN1, EYS, NTNG2, NDNF, NELL2, and EGFLAM."
    ),
    ("ecm_glycoproteins", "NMF3"): (
        "Placental-stromal remodeling glycoprotein ECM",
        "High",
        "Stromal remodeling glycoprotein module enriched in placenta, lung, ovary, gallbladder, urinary bladder, endometrium, and smooth muscle."
    ),
    ("ecm_glycoproteins", "NMF4"): (
        "Specialized sensory-reproductive glycoprotein ECM",
        "Moderate",
        "Specialized glycoprotein module enriched in testis, pituitary gland, ovary, thyroid gland, basal ganglia, and fallopian tube."
    ),
    ("ecm_glycoproteins", "NMF5"): (
        "Vascular-connective glycoprotein ECM",
        "High",
        "Vascular and connective glycoprotein module enriched in blood vessel, heart muscle, adipose tissue, tongue, skeletal muscle, and breast."
    ),
    ("ecm_glycoproteins", "NMF6"): (
        "Epithelial-basement membrane glycoprotein ECM",
        "High",
        "Epithelial and basement membrane glycoprotein module enriched in kidney, vagina, prostate, cervix, skin, fallopian tube, lung, and endometrium."
    ),
    ("ecm_glycoproteins", "NMF7"): (
        "Lymphoid-endocrine glycoprotein ECM",
        "Moderate",
        "Glycoprotein module enriched in choroid plexus, parathyroid gland, spleen, thymus, lymph node, pituitary gland, thyroid gland, and tonsil."
    ),
    ("ecm_glycoproteins", "NMF8"): (
        "Hepatic plasma glycoprotein ECM",
        "High",
        "Hepatic and plasma-associated glycoprotein module enriched in liver, spleen, pancreas, adrenal gland, cerebellum, kidney, and adipose tissue."
    ),
    ("ecm_glycoproteins", "NMF9"): (
        "CNS glycoprotein ECM",
        "Very high",
        "Strong CNS glycoprotein module enriched in cerebral cortex, amygdala, hippocampal formation, midbrain, basal ganglia, hypothalamus, spinal cord, and cerebellum."
    ),
    ("ecm_glycoproteins", "NMF10"): (
        "Reproductive-secretory glycoprotein ECM",
        "Moderate",
        "Reproductive and secretory glycoprotein module enriched in epididymis, seminal vesicle, prostate, liver, vagina, rectum, stomach, and placenta."
    ),

    # Proteoglycans
    ("proteoglycans", "NMF1"): (
        "Epithelial-stromal proteoglycan ECM",
        "High",
        "Proteoglycan module enriched in vagina, skin, salivary gland, blood vessel, prostate, cervix, esophagus, and breast."
    ),
    ("proteoglycans", "NMF2"): (
        "CNS proteoglycan ECM",
        "Very high",
        "Strong CNS proteoglycan module enriched in cerebral cortex, hippocampal formation, amygdala, basal ganglia, midbrain, hypothalamus, cerebellum, and spinal cord."
    ),
    ("proteoglycans", "NMF3"): (
        "Renal-immune stromal proteoglycan ECM",
        "Moderate-high",
        "Proteoglycan module enriched in kidney, lung, lymph node, thyroid gland, tonsil, spleen, and thymus."
    ),
    ("proteoglycans", "NMF4"): (
        "Reproductive-interstitial proteoglycan ECM",
        "High",
        "Interstitial proteoglycan module enriched in gallbladder, fallopian tube, breast, tongue, colon, ovary, smooth muscle, and endometrium."
    ),
    ("proteoglycans", "NMF5"): (
        "Vascular-connective proteoglycan ECM",
        "High",
        "Vascular and connective proteoglycan module enriched in blood vessel, adipose tissue, heart muscle, lung, liver, and smooth muscle."
    ),
    ("proteoglycans", "NMF6"): (
        "GI-mucosal proteoglycan ECM",
        "High",
        "Mucosal and digestive proteoglycan module enriched in placenta, rectum, duodenum, small intestine, stomach, colon, and esophagus."
    ),
    ("proteoglycans", "NMF7"): (
        "Hepatic-digestive proteoglycan ECM",
        "Moderate",
        "Hepatic and digestive proteoglycan module enriched in liver, pancreas, duodenum, salivary gland, small intestine, and skeletal muscle."
    ),
    ("proteoglycans", "NMF8"): (
        "Endocrine-reproductive proteoglycan ECM",
        "Moderate",
        "Endocrine and reproductive proteoglycan module enriched in parathyroid gland, prostate, seminal vesicle, epididymis, adrenal gland, endometrium, and ovary."
    ),
    ("proteoglycans", "NMF9"): (
        "Retinal proteoglycan ECM",
        "Very high",
        "Retinal proteoglycan module enriched in retina and supported by NYX, OPTC, IMPG2, IMPG1, SPOCK1, SPOCK2, and BCAN."
    ),
    ("proteoglycans", "NMF10"): (
        "Hematopoietic-immune proteoglycan ECM",
        "High",
        "Hematopoietic and immune proteoglycan module enriched in bone marrow, thymus, spleen, appendix, placenta, lymph node, and liver."
    ),

    # Collagens
    ("collagens", "NMF1"): (
        "Interstitial collagen-rich stromal ECM",
        "High",
        "Broad interstitial collagen module enriched in gallbladder, ovary, lung, parathyroid gland, spleen, blood vessel, pancreas, and salivary gland."
    ),
    ("collagens", "NMF2"): (
        "CNS-associated collagen ECM",
        "High",
        "Neural-associated collagen module enriched in amygdala, midbrain, hippocampal formation, spinal cord, basal ganglia, cerebral cortex, and hypothalamus."
    ),
    ("collagens", "NMF3"): (
        "Epithelial collagen basement membrane ECM",
        "High",
        "Epithelial collagen module enriched in skin, salivary gland, esophagus, vagina, small intestine, colon, breast, and cervix."
    ),
    ("collagens", "NMF4"): (
        "Renal-muscle basement membrane collagen ECM",
        "Moderate-high",
        "Basement membrane collagen module enriched in thyroid gland, choroid plexus, kidney, heart muscle, skeletal muscle, tongue, and lung."
    ),
    ("collagens", "NMF5"): (
        "Lymphoid collagen remodeling ECM",
        "Moderate",
        "Collagen module enriched in cerebellum, tonsil, lymph node, spleen, urinary bladder, appendix, and thymus."
    ),
    ("collagens", "NMF6"): (
        "Placental-interstitial collagen ECM",
        "High",
        "Interstitial collagen module enriched in placenta, adipose tissue, smooth muscle, endometrium, appendix, fallopian tube, colon, and heart muscle."
    ),
    ("collagens", "NMF7"): (
        "Specialized retinal-reproductive collagen ECM",
        "Moderate",
        "Specialized collagen module enriched in pituitary gland, epididymis, retina, stomach, testis, salivary gland, hypothalamus, and fallopian tube."
    ),
    ("collagens", "NMF8"): (
        "Hepatic-renal collagen ECM",
        "Moderate",
        "Collagen module enriched in liver, kidney, cerebellum, pancreas, thyroid gland, adrenal gland, adipose tissue, and breast."
    ),
    ("collagens", "NMF9"): (
        "Retinal-neural epithelial collagen ECM",
        "Moderate",
        "Collagen module enriched in retina, colon, hippocampal formation, cerebral cortex, cervix, basal ganglia, urinary bladder, and vagina."
    ),
    ("collagens", "NMF10"): (
        "Reproductive-epithelial basement membrane collagen ECM",
        "High",
        "Basement membrane collagen module enriched in blood vessel, cervix, seminal vesicle, endometrium, urinary bladder, skin, prostate, and vagina."
    ),
}


def main() -> None:
    if not BASE_PATH.exists():
        raise FileNotFoundError(f"Combined NMF module summary not found: {BASE_PATH}")

    df = pd.read_csv(BASE_PATH)

    records = []

    for _, row in df.iterrows():
        key = (row["feature_set"], row["component"])

        module_name, confidence, short_interpretation = ANNOTATIONS.get(
            key,
            ("Unannotated module", "Low", "No curated interpretation available.")
        )

        records.append(
            {
                "dataset": row["dataset"],
                "feature_set": row["feature_set"],
                "component": row["component"],
                "module_name": module_name,
                "confidence": confidence,
                "short_interpretation": short_interpretation,
                "top_samples": row["top_samples"],
                "top_genes": row["top_genes"],
            }
        )

    annotated = pd.DataFrame(records)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    annotated.to_csv(OUTPUT_PATH, index=False)

    print(f"[SAVED] {OUTPUT_PATH}")
    print(annotated.head())


if __name__ == "__main__":
    main()