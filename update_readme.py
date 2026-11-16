"""Injecte les tableaux de results/ dans README.md entre les balises <!--TAG-->...<!--/TAG-->."""
import re, pathlib

readme = pathlib.Path("README.md")
text = readme.read_text(encoding="utf-8")

blocks = {
    "REG_TABLE": "results/resultats_regression.md",
    "CLF_TABLE": "results/resultats_classification.md",
    "CV_TABLE": None,
    "GRID_TABLE": None,
}

import pandas as pd
blocks["CV_TABLE"] = pd.read_csv("results/validation_croisee_rf.csv").to_markdown(index=False)
blocks["GRID_TABLE"] = pd.read_csv("results/gridsearch_meilleurs_parametres.csv").to_markdown(index=False)

for tag, src in blocks.items():
    content = src if (src and not src.endswith(".md")) else pathlib.Path(src).read_text(encoding="utf-8") if src else ""
    pattern = re.compile(rf"(<!--{tag}-->)(.*?)(<!--/{tag}-->)", re.S)
    text = pattern.sub(lambda m: f"{m.group(1)}\n\n{content}\n\n{m.group(3)}", text)

readme.write_text(text, encoding="utf-8")
print("README.md mis à jour avec vos résultats.")
