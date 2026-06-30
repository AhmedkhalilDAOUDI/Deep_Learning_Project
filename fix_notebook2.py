
path = r"C:\Users\Khalil\Desktop\Deep Learning project\consignes_projet_final_classification_images.ipynb"
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# The broken cell block to replace
old_block = '''  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "id": "P_tUVvvxxR3F"
   },
   "outputs": [],
   "source": [
     "# ── Code cell 25 ──\\n",
     "if len(errors) == 0:\\n",
     "    print(\"Aucune erreur sur le jeu de validation : le modèle a prédit toutes les images correctement.\")\\n",
     "else:\\n",
     "    print(f\"Nombre d'erreurs: {len(errors)} / {len(y_test)}\")\\n",
     "\\n",
     "    # Récupérer les caractéristiques des images erronées\\n",
     "    error_paths = filepaths[errors]\\n",
     "    error_true = y_test[errors]\\n",
     "    error_pred = y_pred_class[errors]\\n",
     "\\n",
     "    from PIL import ImageOps\\n",
     "\\n",
     "    patterns = {\\n",
     "        \\"petite_taille\\": 0,\\n",
     "        \\"bg_clair\\": 0,\\n",
     "        \\"bg_sombre\\": 0,\\n",
     "        \\"pose_rare\\": 0,\\n",
     "        \\"cadrage_serre\\": 0,\\n",
     "        \\"cadrage_large\\": 0,\\n",
     "    }\\n",
     "\\n",
     "    details = []\\n",
     "    for fp, t, p in zip(error_paths, error_true, error_pred):\\n",
     "        img = Image.open(fp).convert(\\"RGB\\")\\n",
     "        w, h = img.size\\n",
     "        pixels = np.array(img)\\n",
     "        brightness = pixels.mean()\\n",
     "\\n",
     "        # Règles simples pour détecter des patterns\\n",
     "        if min(w, h) < 100:\\n",
     "            patterns[\\"petite_taille\\"] += 1\\n",
     "        if brightness > 200:\\n",
     "            patterns[\\"bg_clair\\"] += 1\\n",
     "        if brightness < 50:\\n",
     "            patterns[\\"bg_sombre\\"] += 1\\n",
     "        if max(w, h) / min(w, h) > 1.8:\\n",
     "            patterns[\\"cadrage_serre\\"] += 1\\n",
     "        if max(w, h) / min(w, h) < 1.1:\\n",
     "            patterns[\\"cadrage_large\\"] += 1\\n",
     "        details.append((fp, t, p, w, h, brightness))\\n",
     "\\n",
     "    print(\\"\\n=== Patterns communs sur les images mal classées ===\\")\\n",
     "    for k, v in patterns.items():\\n",
     "        if v:\\n",
     "            print(f\\"- {k}: {v} cas ({v/len(errors):.0%})\\")\\n",
     "\\n",
     "    print(\\"\\nObservations qualitatives :\\")\\n",
     "    print(\\"- Les images avec fond très clair (ex: neige, sol blanc) ou très sombre peuvent induire le modèle.\\")\\n",
     "    print(\\"- Les chiens/chats de petite taille ou en mouvement rapide sont plus souvent mal classés.\\")\\n",
     "    print(\\"- Certaines races atypiques ou des positions couchées/allongées créent des confusions.\\")\\n",
     "        print(\\"- Les zooms\\n",
     "    excessive\\n",
     "s ou les images floues réduisent la précision des features.\\n",
     "    print(\\"- En résumé, les erreurs sont souvent corrélées à un contexte visuel ambigu ou une qualité d'image réduite.\\")\\n"
   ]
  },'''

# Actually wait, looking at the current file, the source lines have some corruption.
# Let me first read the current state of the file to see the exact lines.
