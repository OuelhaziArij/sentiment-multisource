# Explication pour la soutenance (français)

## 1. Problème
Une plateforme e-commerce reçoit des avis sous trois formes : texte, capture d'écran, message vocal. Il faut prédire le sentiment (positif / négatif / neutre) quelle que soit la forme, et l'exposer via une API.

## 2. Architecture et pourquoi une cascade
Image → OCR → texte ; audio → ASR → texte ; puis **un seul** classifieur de texte. Avantages : un seul modèle à entraîner, évaluer et maintenir ; les modules OCR/ASR s'améliorent indépendamment ; l'API est homogène (même JSON). Inconvénients : les erreurs d'OCR/ASR se propagent au classifieur (pas de rattrapage), et on perd l'information non textuelle (ton de la voix, mise en forme, étoiles non lues). Alternative : modèles multimodaux de bout en bout, mais il faut beaucoup plus de données.

## 3. Choix techniques
- **Nettoyage minimal** : minuscules, suppression des accents (robustesse OCR/ASR), ponctuation et chiffres retirés, stop-words retirés **sauf négations** (« pas satisfait » ≠ « satisfait »).
- **TF-IDF** : pondère chaque mot par sa fréquence dans le document (TF, sous-linéaire) × sa rareté dans le corpus (IDF). **Mots (1,2) + caractères (2,5)** : les n-grammes de caractères tolèrent les fautes d'OCR/ASR.
- **Régression Logistique** : modèle linéaire, un score par classe (softmax), `predict_proba` donne la confiance ; régularisation L2 (C=10). Choisie plutôt que LinearSVC car elle fournit des probabilités (LinearSVC n'en donne pas sans calibration, impraticable sur 24 exemples).
- **OCR** : Pillow (niveaux de gris, autocontraste, agrandissement ×2, netteté) puis Tesseract `fra` (`--psm 6`, bloc de texte) ; Tesseract segmente l'image, reconnaît les caractères avec un réseau LSTM et un dictionnaire de langue. Suppression de la bannière parasite par expressions régulières.
- **Whisper** : transformeur encodeur-décodeur entraîné sur 680 000 h d'audio ; l'audio est converti par ffmpeg (mp3/wav → 16 kHz mono), découpé en fenêtres de 30 s, transformé en spectrogramme log-Mel, l'encodeur le résume et le décodeur génère le texte (langue forcée `fr`). Chargement **paresseux** (lent et gourmand) et taille choisie via `WHISPER_MODEL`.
- **FastAPI** : routes typées, Pydantic valide automatiquement les entrées/sorties, Swagger généré (`/docs`). Endpoints synchrones (`def`) exécutés dans un pool de threads pour ne pas bloquer la boucle d'événements pendant OCR/ASR.
- **Validation** : MIME déclaré puis repli sur l'extension, signature binaire (magic bytes) pour détecter l'audio corrompu, lecture par blocs avec arrêt à la limite (413), fichier vide / image illisible / texte blanc → 422, outil indisponible → 503.
- **Tests** : 53 tests pytest ; Whisper simulé (rapide, déterministe) ; OCR réel sur les images du dataset.

## 4. Résultats (tous issus du code exécuté)
- Données : 24 textes (8/8/8), 9 images, 9 audios.
- **CV répétée sur 24 textes : accuracy 0,27 ± 0,16 — SOUS le hasard (0,33)**. Raison : vocabulaire quasi disjoint entre textes + effet de la validation croisée sur classes parfaitement équilibrées. Ce n'est pas un bug (LOO = 0,21, reproductible).
- **Test externe (18 textes de référence OCR/ASR)** : 0,833 (15/18), macro-F1 0,83. IC95 ≈ ±17 points ; textes du même gabarit → optimiste.
- CV poolée (42 textes) : ≈ 0,69 ; baseline 0,29.
- **OCR** : similarité 0,996, CER 0,9 % ; sentiment sur texte OCR réel 7/9 (0,78).
- **Audio** : décodage ffmpeg 9/9 ; **Whisper non exécuté dans mon environnement (poids inaccessibles) : aucune métrique ASR** ; classifieur sur transcriptions de référence 8/9.
- Erreurs : I01 (étoiles non lues, « sans hésiter » ignoré), I09 (neutre jugé négatif), A05. Confiance faible (moyenne 0,60), non calibrée.

## 5. Limites à assumer
Très peu de données ; ne pas affirmer une bonne généralisation ; test externe issu du même gabarit ; ASR non validé de bout en bout ici ; OCR testé sur captures propres seulement ; filtre parasite spécifique au jeu fourni ; le fait que les entêtes (« RÉCLAMATION », « 10/10 ») soient dans la référence aide le classifieur.

## 6. Dix questions probables du jury
1. **Pourquoi une cascade ?** Un seul classifieur, modules indépendants, API homogène ; coût : propagation des erreurs OCR/ASR.
2. **Pourquoi garder les négations ?** Elles inversent le sens ; les retirer ferait confondre « pas satisfait » et « satisfait ».
3. **TF-IDF, c'est quoi ?** Fréquence du mot dans le document × log(inverse de sa fréquence documentaire) : valorise les mots distinctifs.
4. **Pourquoi la régression logistique ?** Simple, adaptée aux petites données, probabilités natives pour `confiance`, interprétable.
5. **Pourquoi la CV donne 0,27, sous le hasard ?** 18 textes d'entraînement par pli, vocabulaire quasi disjoint, classes équilibrées qui se « compensent » dans le pli ; signale que 24 exemples sont insuffisants, d'où le test externe et ma prudence.
6. **Peut-on faire confiance au 0,83 ?** Avec réserve : 18 exemples (IC ±17 pts), même gabarit que l'entraînement.
7. **Que veut dire `confiance` ?** Le max de `predict_proba`, non calibré ; sert de signal relatif (0,65 si juste, 0,39 si faux sur les images).
8. **Comment l'OCR gère-t-il le bruit ?** Prétraitement Pillow, n-grammes de caractères, normalisation sans accents ; filtre du texte parasite ; mesuré : CER 0,9 %.
9. **Pourquoi Whisper en chargement paresseux ?** Chargement long et mémoire importante ; on ne le paie qu'à la première requête audio ; taille via `WHISPER_MODEL` (compromis vitesse/précision).
10. **Comment sont gérés les fichiers invalides ?** MIME → extension → signature ; 422 (vide, corrompu, type refusé), 413 (taille), 503 (service OCR/ASR indisponible) ; testés par pytest.

Bonus : *Et si Whisper se trompe ?* Le texte erroné est classifié tel quel ; le champ `texte_extrait` permet de le voir et la confiance basse alerte.
