# Planning - Resume des plots de evaluate.py

## Portee
Ce resume couvre les graphiques generes par src/caloch_eval/evaluate.py, principalement via src/caloch_eval/evaluate_plotting_helper.py et via les methodes de HighLevelFeatures.

## Conditions globales de generation
- Les douches generees et de reference sont coupees avec args.cut avant calculs/plots.
- Les high-level features sont calculees avant les plots histogrammes.
- Le fichier de metriques histogram_chi2_<dataset>.txt est initialise en mode hist-chi/hist/all/no-cls puis enrichi par plusieurs fonctions de plotting.

## Plots mode avg / no-cls / all
- Comparaison couche par couche reference vs genere
  - Fonction: plot_layer_comparison
  - Construction:
    - Parcourt reference_class.relevantLayers.
    - Decoupe chaque couche avec reference_class.bin_edges.
    - Trace 2 panneaux (reference a gauche, genere a droite) avec _DrawSingleLayer.
    - Utilise un vmax commun base sur la reference.
  - Sortie: Average_Layer_<layer_id>_dataset_<dataset>.pdf (un PDF par couche).

- Moyenne des douches (genere)
  - Fonction: hlfs[0].DrawAverageShower
  - Construction:
    - Moyenne de showers[0] sur l'axe evenements.
  - Sortie: average_shower_dataset_<dataset>.pdf.

- Moyenne des douches (reference)
  - Fonction: hlfs[0].DrawAverageShower
  - Construction:
    - Moyenne de reference_shower sur l'axe evenements.
    - Mise en cache dans reference_hlf.avg_shower.
  - Sortie: reference_average_shower_dataset_<dataset>.pdf.

- Douches individuelles (5 premieres)
  - Fonction: hlfs[0].DrawSingleShower
  - Construction:
    - Trace showers[0][:5] et reference_shower[:5].
  - Sorties:
    - single_shower_dataset_<dataset>.pdf
    - reference_single_shower_dataset_<dataset>.pdf

## Plots mode avg-E / no-cls / all
- Moyenne des douches par intervalle d'energie
  - Fonction: hlfs[0].DrawAverageShower
  - Construction:
    - DS1: target_energies = valeurs uniques triees de reference_energy.
    - DS2/DS3: bornes logspace 10^3..10^6.
    - Pour chaque intervalle [E_i, E_{i+1}):
      - calcule le masque d'evenements generes et reference,
      - trace la moyenne generee,
      - trace la moyenne reference (cache reference_hlf.avg_shower_E[E_i]).
  - Sorties:
    - average_shower_dataset_<dataset>_E_<E_i>.pdf
    - reference_average_shower_dataset_<dataset>_E_<E_i>.pdf

## Plots mode hist-p / hist-chi / hist / no-cls / all
Ces plots passent par plot_histograms puis, pour DS1, plot_atlas_style est appele en plus dans main.

- Etot/Einc global
  - Fonction: plot_Etot_Einc
  - Construction:
    - Histogramme GEANT en panneau haut + bande d'erreur.
    - Pour chaque modele: histogramme + bande d'erreur.
    - Panneau bas: ratio modele/GEANT avec lignes de repere 0.7, 1.0, 1.3.
  - Sortie: Etot_Einc_dataset_<dataset>.pdf.
  - Metrique: separation power ecrite dans histogram_chi2_<dataset>.txt.

- Energie deposee par couche
  - Fonction: plot_E_layers
  - Construction:
    - Un page/panneau par couche via PdfPages.
    - Bins log si arg.x_scale == log (avec ajustement quantile pour energy label).
    - Panneau haut densite, panneau bas ratio modele/GEANT.
  - Sortie: E_layer_dataset_<dataset>.pdf.
  - Metrique: separation power par couche.

- Centre d'energie en eta par couche
  - Fonction: plot_ECEtas
  - Construction:
    - Limites d'axe dependantes du dataset/couche (ou quantiles si energy est fourni).
    - Panneau haut densite, panneau bas ratio.
  - Sortie: ECEta_layer_dataset_<dataset>.pdf.
  - Metrique: separation power par couche.

- Centre d'energie en phi par couche
  - Fonction: plot_ECPhis
  - Construction: meme structure que ECEtas.
  - Sortie: ECPhi_layer_dataset_<dataset>.pdf.
  - Metrique: separation power par couche.

- Largeur du centre d'energie en eta
  - Fonction: plot_ECWidthEtas
  - Construction: histogrammes + ratio par couche, limites fixes ou quantiles.
  - Sortie: WidthEta_layer_dataset_<dataset>.pdf.
  - Metrique: separation power par couche.

- Largeur du centre d'energie en phi
  - Fonction: plot_ECWidthPhis
  - Construction: histogrammes + ratio par couche, limites fixes ou quantiles.
  - Sortie: WidthPhi_layer_dataset_<dataset>.pdf.
  - Metrique: separation power par couche.

- Sparsite
  - Fonction: plot_sparsity
  - Construction:
    - Trace 1 - sparsite par couche.
    - Panneau haut densite, panneau bas ratio.
  - Sortie: Sparsity_layer_dataset_<dataset>.pdf.
  - Metrique: separation power par couche.

- Distribution d'energie voxel globale
  - Fonction: plot_cell_dist
  - Construction:
    - Histogramme global des voxels reference vs modeles.
    - Y en log; X en log si arg.x_scale == log.
    - Panneau bas ratio modele/GEANT.
  - Sortie: voxel_energy_dataset_<dataset>.pdf.
  - Metrique: separation power.

## Plots specifiques DS1
- Etot/Einc discret par energie incidente
  - Fonction: plot_Etot_Einc_discrete
  - Construction:
    - Pagination PdfPages en grilles 4x4.
    - 15 histogrammes max par page + panneau de legende.
    - Bins resserres pour photons aux hautes energies.
  - Sortie: Etot_Einc_dataset_<dataset>_E_i.pdf.
  - Metrique: separation power par intervalle energetique.

- Atlas style multi-energie
  - Fonction: plot_atlas_style
  - Construction:
    - Bins dynamiques a partir de reference_class.Einc.
    - Pagination PdfPages, style ATLAS 8x5 par page.
    - Panneaux histogramme + ratio par energie, avec bande d'incertitude reference.
  - Sortie: Etot_Einc_dataset_<dataset>_E_i.pdf.
  - Remarque:
    - Meme nom de sortie que plot_Etot_Einc_discrete, donc ecrasement possible selon l'ordre d'appel.

## Classifieur
- Les modes cls-low / cls-low-normed / cls-high / all n'ajoutent pas de figures explicites dans evaluate.py.
- Ils produisent des fichiers texte de resultats: classifier_<mode>_<dataset>_input_<n>.txt.

## References code
- Entrees et modes: src/caloch_eval/evaluate.py
- Orchestration histogrammes: src/caloch_eval/evaluate.py (plot_histograms et bloc main)
- Implementations des figures: src/caloch_eval/evaluate_plotting_helper.py
