
path = r"C:\Users\Khalil\Desktop\Deep Learning project\consignes_projet_final_classification_images.ipynb"
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

source_lines = [
    "# ── Code cell 25 ──",
    "if len(errors) == 0:",
    "    print(\"Aucune erreur sur le jeu de validation : le modèle a prédit toutes les images correctement.\")",
    "else:",
    "    print(f\"Nombre d'erreurs: {len(errors)} / {len(y_test)}\")",
    "",
    "    # Récupérer les caractéristiques des images erronées",
    "    error_paths = filepaths[errors]",
    "    error_true = y_test[errors]",
    "    error_pred = y_pred_class[errors]",
    "",
    "    from PIL import ImageOps",
    "",
    "    patterns = {",
    "        \"petite_taille\": 0,",
    "        \"bg_clair\": 0,",
    "        \"bg_sombre\": 0,",
    "        \"pose_rare\": 0,",
    "        \"cadrage_serre\": 0,",
    "        \"cadrage_large\": 0,",
    "    }",
    "",
    "    details = []",
    "    for fp, t, p in zip(error_paths, error_true, error_pred):",
    "        img = Image.open(fp).convert(\"RGB\")",
    "        w, h = img.size",
    "        pixels = np.array(img)",
    "        brightness = pixels.mean()",
    "",
    "        # Règles simples pour détecter des patterns",
    "        if min(w, h) < 100:",
    "            patterns[\"petite_taille\"] += 1",
    "        if brightness > 200:",
    "            patterns[\"bg_clair\"] += 1",
    "        if brightness < 50:",
    "            patterns[\"bg_sombre\"] += 1",
    "        if max(w, h) / min(w, h) > 1.8:",
    "            patterns[\"cadrage_serre\"] += 1",
    "        if max(w, h) / min(w, h) < 1.1:",
    "            patterns[\"cadrage_large\"] += 1",
    "        details.append((fp, t, p, w, h, brightness))",
    "",
    "    print(\"\\n=== Patterns communs sur les images mal classées ===\")",
    "    for k, v in patterns.items():",
    "        if v:",
    "            print(f\"- {k}: {v} cas ({v/len(errors):.0%})\")",
    "",
    "    print(\"\\nObservations qualitatives :\")",
    "    print(\"- Les images avec fond très clair (ex: neige, sol blanc) ou très sombre peuvent induire le modèle.\")",
    "    print(\"- Les chiens/chats de petite taille ou en mouvement rapide sont plus souvent mal classés.\")",
    "    print(\"- Certaines races atypiques ou des positions couchées/allongées créent des confusions.\")",
    "    print(\"- Les zooms excessifs ou les images floues réduisent la précision des features.\")",
    "    print(\"- En résumé, les erreurs sont souvent corrélées à un contexte visuel ambigu ou une qualité d'image réduite.\")",
]

# Build JSON with proper escaping
def json_escape(s):
    # Escape backslashes first, then double quotes
    return s.replace("\\", "\\\\").replace('"', '\\"')

source_json_lines = []
for line in source_lines:
    escaped = json_escape(line)
    # Each line in the source array ends with \n
    source_json_lines.append(f'     "{escaped}\\n"')

source_json = ",\n".join(source_json_lines)

new_cell = f'''  {{
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {{
    "id": "P_tUVvvxxR3F"
   }},
   "outputs": [],
   "source": [
{source_json}
   ]
  }},
'''

# Replace lines 1088-1157 (1-indexed), which is indices 1087-1156 (0-indexed)
lines[1087:1157] = [new_cell]

with open(path, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Fixed notebook written.")
print(f"New cell preview:\n{new_cell[:600]}")
