# Rapport détaillé : capteurs de force des pieds du RHPS1 (2026-11-04)

Résumé court en anglais : [2026-11-04-force-sensors.md](2026-11-04-force-sensors.md). Ce document donne le protocole, toutes les mesures et les commandes pour refaire chaque étape.

## 0. En une page

**Problème de départ.** Avec le contrôleur d'un collègue (autre PC), le robot s'affaisse vers la droite. On a soupçonné d'abord le changement de gain `KI[2]` des chevilles (2.0 -> 0.02, fait le 2026-10-29), puis les capteurs de force.

**Ce qu'on a établi**

| # | Constat | Niveau de confiance |
|---|---|---|
| 1 | Le capteur du pied droit est juste (rapport capteur / balance = 1,00 +- 0,01 sur 3 charges). | Bon : 3 points, très linéaire |
| 2 | Le capteur du pied gauche lit environ **12 % trop haut** (rapport 1,09 à 1,13 sur 4 charges), erreur d'échelle pure (pas de décalage). | Bon : 4 points + répété une seconde fois |
| 3 | Une fois le gauche corrigé, le robot est **réellement plus chargé à droite** : 60 / 40 (écart 100 N) dans la posture task d'origine (genou 35,6 deg). | Bon : stable et répété, mais la correction gauche n'est pas encore revérifiée à la balance |
| 4 | Cet écart **diminue quand on plie les genoux** : 52,7 / 47,3 (genou 49 deg), 51,9 / 48,1 (genou 71,4 deg). | Moyen : un seul essai par posture |
| 4b | Le **centre de masse est quasi centré** (environ 4 mm d'après l'équilibre des moments) : l'écart est surtout porté par les moments sous les pieds, pas par une répartition de masse (une estimation de 2 cm faite plus tôt était fausse, voir §5.1). | Moyen : 1 run, 3 points, hypothèse sur `Mx` |
| 5 | Les mesures au sol sont stables (< 0,4 % sur 4 min). Le **zéro** du capteur gauche dérive d'environ **+9 N en 50 min** (robot en l'air). Le droit reste dans +-3 N. | Bon : enregistrement de 3600 s |
| 6 | `KI[2]` des 4 chevilles a été remis à **2.0** et sauvegardé dans les drives. Il n'est plus soupçonné d'être la cause (non testé en marche). | Bon pour l'état des drives |

**Ce qu'on n'a pas établi**

- La **cause** du défaut du capteur gauche (calibration ATI, `CountPerForce` mal lu, capteur différent).
- La **cause** de l'asymétrie de charge. Hypothèse de travail (non mesurée) : une petite différence de hauteur effective entre les pieds (offset d'articulation de ~0,1 deg, semelle, sol), amplifiée par la raideur des jambes tendues (§5.4-5.5). L'IMU n'a pas la résolution pour la voir.
- Si l'erreur du capteur gauche explique la chute à droite avec le contrôleur du collègue.

## 1. Matériel et outils

- Robot RHPS1, quatre capteurs F/T ATI EtherCAT, d'après `RHPS1.conf` :

| `sensorId` | Nom EtherCAT | Position |
|---|---|---|
| 0 | `Term RF (ATI EtherCAT F/T Sensor)` | pied droit |
| 1 | `Term LF (ATI EtherCAT F/T Sensor)` | pied gauche |
| 2 | `Term RH (ATI EtherCAT F/T Sensor)` | main droite |
| 3 | `Term LH (ATI EtherCAT F/T Sensor)` | main gauche |

  Les capteurs de pied sont entre le mollet et le pied (le poids du pied n'est pas dans les efforts, mais il est tarré si le zéro est fait robot en l'air).
- `io_main` : processus de bas niveau (IOB, EtherCAT). Il remplit `state.force[sensorId][6]` (Fx, Fy, Fz, Mx, My, Mz) en mémoire partagée.
- `bodystat` : affiche l'état du robot (angles, courants, forces, accéléromètre) depuis cette mémoire partagée.
- `sdo` : lit ou écrit un objet SDO sur un driver moteur ELMO via `io_main`.
- Contrôleur de la session : `HalfSitPose` (mc_rtc), qui tient une posture. Les logs mc_rtc sont des `.bin` dans `/tmp`.
- Une balance (kg) sous un pied, et la masse du robot : **58,24 kg** (570,7 N avec g = 9,81).

**Précautions de lecture avec `bodystat`**

- Les forces affichées sont lissées par un filtre qui repart de zéro à chaque lancement : attendre environ 2 s (une dizaine d'images) et lire la **dernière** image.
- Les colonnes `torque(%)` sont en réalité des **courants moteur en ampères** (voir `bodystat.cpp` local). Le nombre entre parenthèses n'est pas identifié.
- Ordre des lignes de forces : RF, LF, RH, LH (ordre des `sensorId`).

Lecture type :

```
timeout 4 bodystat > bs.raw
sed 's/\x1b\[[0-9;]*[A-Za-z]//g' bs.raw > bs.txt
# dernière image : les 4 lignes sous "Force sensor"
```

## 2. Chronologie et mesures

### 2.1 Gains des chevilles (avant la session)

- 2026-10-29 : `KI[2]` des 4 chevilles `LAnkleRPI`, `LAnkleRPO`, `RAnkleRPI`, `RAnkleRPO` mis à 0.02 (valeur d'origine 2.0), puis sauvegardé. `KP[3]=60` et `GS[2]=0` inchangés.
- Entre le 2026-11-02 et le 2026-11-04 : remis à 2.0 et sauvegardé, pour éliminer ce paramètre comme cause (voir §6).

### 2.2 Premières lectures de forces

Robot servos éteints, `io_main` sans contrôleur.

- Robot non posé en posture task (état exact de l'appui non noté), avant zéro : RF Fz = 2,8 à 3,8 N, LF Fz = 230 N (après stabilisation du filtre), mains -11,1 N chacune. Valeurs non exploitables : le zéro n'était pas fait et l'appui n'est pas connu.
- Après un zéro fait en l'air par l'utilisateur : RF Fz = 0,48 N, LF = 0,38 N, mains -0,01 et -0,06 N. Le zéro est bon.
- Posé en posture task, sans le gain de §4 :

| Placement | RF Fz | LF Fz | Somme | Droite |
|---|---|---|---|---|
| 1er | 314,7 N | 221,7 N | 536,4 N | 58,7 % |
| 2e | 304,2 N | 231,2 N | 535,4 N | 56,8 % |

  Le déséquilibre se reproduit, mais varie d'une dizaine de newtons selon le placement.
- Remarque : la somme (535 N, 54,5 kg) est inférieure à la masse du robot (58,24 kg = 571 N). Ce n'est pas expliqué (voir §7).

### 2.3 Contrôle d'échelle à la balance

**Protocole (par l'utilisateur).** Le robot est posé sur une seule jambe, le pied sur la balance. On note la masse de la balance et la ligne du capteur correspondant dans `bodystat`. Trois à quatre charges par côté (en variant la charge, par la hauteur de la jambe ou les appuis).

Force attendue = masse x 9,81. Rapport = Fz du capteur / force attendue.

| Côté | Balance | Attendue | Fz capteur | Rapport |
|---|---|---|---|---|
| Droit | 14,38 kg | 141,1 N | 142 N | **1,007** |
| Droit | 28 kg | 274,7 N | 275 N | **1,001** |
| Droit | 55 kg | 539,6 N | 533 N | **0,988** |
| Gauche | 15,9 kg | 156,0 N | 174 N | **1,115** |
| Gauche | 17,5 kg | 171,7 N | 194 N | **1,130** |
| Gauche | 22 kg | 215,8 N | 236 N | **1,094** |
| Gauche | 34 kg | 333,5 N | 377 N | **1,130** |

Ajustement linéaire : droit pente ~0,98 (décalage +3,6 N), gauche pente ~1,14 (décalage ~-4,6 N). L'utilisateur a répété la mesure plus tard : « toujours 12 % ».

Mesures horizontales relevées (Fx, Fy, Fz) :

- Droit : (-2,7 ; -12,8 ; 142), (1,9 ; 33,5 ; 275), (25 ; 39 ; 533).
- Gauche : (1,4 ; 0,8 ; 174), (0,5 ; 2 ; 194), (0,5 ; 2,5 ; 236), (-2,7 ; 0 ; 377).

**Pourquoi ce n'est ni le poids des pieds ni l'inclinaison**

- Le poids d'un pied est un décalage constant : il déplace la droite, pas sa pente. Les pentes sont calculées entre points, donc insensibles à lui.
- Si la jambe est inclinée de theta, le capteur lit N cos(theta) : **moins** que la balance, jamais plus. À gauche, Fx et Fy sont presque nuls (jambe droite). À droite, Fy/Fz atteint 0,07 (environ 4 deg), soit moins de 0,3 % sur Fz.
- Les trois points droits sont proportionnels à 1 % près : la balance et le protocole sont fiables.

**Limite** : l'identification « côté mesuré = ligne de `bodystat` utilisée » repose sur l'utilisateur. Après correction (§4), le capteur d'`id 1` a bien baissé de 12 % ; cela confirme l'ordre, mais la balance n'a pas été refaite après correction (à faire, §8).

### 2.4 Origine possible du défaut (lecture du code)

Dans `hrp5p-iob` (branche `topic/EC-master3.2-rhps1`) :

- `SoemBodySignalLib/src/DeviceForceSensorImpl.cpp` : `force = valeur_brute / countPerForceData_` (idem couple avec `countPerTorqueData_`).
- `SoemBodySignalLib/src/EcManager.cpp` (lignes ~270-286) : `countPerForceData_` est lu dans chaque capteur par SDO `0x2040:49`, `countPerTorqueData_` par `0x2040:50`, à l'initialisation d'EtherCAT. La valeur par défaut est **1000** si la lecture échoue. La console d'`io_main` affiche `[adresse] CountPerForce = ...`.
- `iob/TransListenerImpl.cpp` (lignes ~136-143) : `state.force = force + offset.force` (offset additif, nul ici). Il n'y a **aucun gain d'échelle** côté IOB avant notre patch.
- Les offsets de forces de `offset.dat` sont tous à 0 ; le fichier date du 2026-07-31.

Donc un défaut d'échelle de 12 % ne vient pas de notre code. Il vient de ce que renvoie le capteur gauche ou de sa configuration de calibration. Les lignes `CountPerForce` de la console n'ont pas pu être relevées (console masquée ; `io_main` tourne en root et `/proc` est refusé).

## 3. Mesures en posture (avant correction logicielle)

Voir le tableau du §2.2. Les efforts horizontaux étaient opposés d'un pied à l'autre (RF Fx +17 N / LF Fx -15 N ; RF Fy +15 N / LF Fy -18 N) : les pieds se poussent l'un contre l'autre, ou le montage des axes diffère.

## 4. Contournement dans l'IOB (gain par capteur)

**Modification** (locale, dans `hrp5p-iob`, non poussée upstream ; fichier `docs/hrp5p-iob_force_gain.patch`) :

- `iob/TransListenerImpl.cpp` / `.hpp` : `force = raw * gain_force + offset`, `torque = raw * gain_torque + offset`.
- Gains lus au démarrage dans `<IOB_SHARE_DIR>/force_gain.dat` (ici `~/sanbox_leo/workspace/install/share/iob/force_gain.dat`), une ligne par `sensorId`. Fichier absent = 1.0.
- `io_main` affiche au démarrage `force sensor N gain: force ... torque ...`.

Contenu utilisé (gain 0,89 = 1 / 1,12 sur la force du pied gauche, rien sur les moments, non mesurés) :

```
1.0 1.0
0.89 1.0
1.0 1.0
1.0 1.0
```

**Déploiement**

```
ninja -C build io_main
cp ~/sanbox_leo/workspace/install/bin/io_main ~/sanbox_leo/workspace/install/bin/io_main.bak-20261104
cp build/iob/io_main ~/sanbox_leo/workspace/install/bin/io_main
```

Il existe deux `io_main` sur le PC : `~/sanbox_leo/workspace/install/bin/` (remplacé) et `~/workspace/install/bin/` (31/07, non modifié). Le nouveau a été relancé à 19:29 (arrêt de l'ancien par l'utilisateur).

**Vérification** : le pied gauche est passé de 231 N à 208 N, le droit est resté à ~310 N.

**Retour arrière** : supprimer `force_gain.dat`, ou remettre `io_main.bak-20261104`.

**À reporter** sur tout autre PC qui lance l'IOB (patch + build + `force_gain.dat`). À retirer si le capteur est réparé à la source.

## 5. Répartition de charge selon la posture (avec le gain)

Posture task d'origine (angles commandés, deg) : `CROTCH_Y` ∓0,6035, `CROTCH_R` ±1,649, `CROTCH_P` -15,538, `KNEE_P` 35,639, `ANKLE_R` ∓1,756, `ANKLE_P` -20,092, buste, tête 0, épaule P 15, épaule R ±10, épaule Y ∓5, coude P -30.

Seuls le genou, la cheville en tangage et le buste ont été changés (symétriquement à gauche et à droite) ; le reste est resté fixe. Le robot était soulevé entre les postures (somme des forces ~0).

Résultats du log `HalfSitPose` (plateaux au sol, moyenne sur la durée indiquée) :

| Posture (commande, deg) | t (s) | Fz D | Fz G | Somme | D | Écart | Fx,Fy D | Fx,Fy G |
|---|---|---|---|---|---|---|---|---|
| genou 35,6, cheville P -20,1, buste 0 | 2-18 | 306,9 | 206,9 | 513,8 | 59,7 % | 100,0 N | 11,7 ; 12,2 | -10,1 ; -15,5 |
| genou 49,0, cheville P -21,1, buste 39 | 260-500 | 275,5 | 247,7 | 523,3 | 52,7 % | 27,8 N | 6,5 ; 8,8 | -5,4 ; -10,6 |
| genou 71,4, cheville P -36,7, buste 21 | 825-860 | 275,2 | 255,5 | 530,7 | 51,9 % | 19,7 N | 0,3 ; -0,3 | -1,7 ; -3,6 |

Moments (Mx, My en N.m) : posture 1 D (0,8 ; 13,3) G (5,0 ; -10,1) ; posture 2 D (-1,1 ; 17,2) G (0,0 ; 1,1) ; posture 3 D (6,5 ; 20,2) G (-6,0 ; 13,5).

Même résultat servos éteints et allumés (posture 1 : 310,3 / 208,3 N éteints, 308,4 / 207,9 N allumés). Erreur de suivi (mesuré moins commandé) : au plus 0,0005 deg aux hanches et chevilles, 0,015 deg aux genoux dans la posture 3. L'accéléromètre donne une inclinaison avant-arrière d'environ 1,8 deg (posture 1), 12 deg (posture 2) et 20 deg (posture 3).

### 5.1 Ce que dit l'équilibre des moments

Pour un robot immobile, la répartition de charge dépend de (a) la position latérale du centre de masse, (b) du point d'appui sous chaque pied (moment `Mx` transmis par la cheville) et (c) de la raideur des jambes. Équilibre des moments en roulis, avec `e` le décalage latéral du centre de masse et `d` l'espacement des pieds :

`W * e = (d / 2) * (Fz_G - Fz_D) + (Mx_D + Mx_G)`

Ajustement aux trois plateaux (hypothèse : `Mx` est le moment de roulis, le signe `+` du terme de moments donne un `d` positif) : **`e` ~ 4 mm** et **`d` ~ 15 cm**, avec des écarts résiduels d'au plus 1,2 N.m (le terme de force vaut jusqu'à 7,7 N.m en posture 1).

| Posture | W (N) | Fz_G - Fz_D (N) | Mx_D + Mx_G (N.m) |
|---|---|---|---|
| 1 genou 35,6 | 514 | -100,0 | 0,84 + 4,96 = 5,80 |
| 2 genou 49 | 523 | -27,8 | -1,08 + 0,02 = -1,06 |
| 3 genou 71,4 | 531 | -19,7 | 6,52 - 6,01 = 0,51 |

**Correction d'une estimation précédente.** Une première estimation donnait environ 2 cm de décalage latéral du centre de masse (100 N d'écart sur 514 N, pieds à 20 cm). Elle ignorait les moments sous les pieds et est **fausse** : l'essentiel du déséquilibre de charge est porté par les moments `Mx` des chevilles, et le centre de masse est quasi centré (quelques millimètres). Le modèle ne repose que sur un run, trois points et une hypothèse d'orientation de `Mx` : ordres de grandeur seulement.

### 5.2 Géométrie de la commande

- Chaîne de tangage (`CROTCH_P + KNEE_P + ANKLE_P`, jambe gauche) : 0,01 deg (posture 1), 12,36 deg (posture 2), 19,16 deg (posture 3). Elle vaut l'inclinaison du bassin : le bassin se penche de ces angles pour garder les semelles à plat en tangage (IMU : 12,6 et 20,3 deg relatifs, voir §5.3).
- Chaîne de roulis (`CROTCH_R + ANKLE_R`) : **-0,107 deg à gauche et +0,107 deg à droite**, identique dans les trois postures. Les semelles ne sont donc pas exactement à plat en roulis dans la commande : chacune est inclinée de 0,107 deg (environ 0,2 mm sur 10 cm de semelle) et repose sur un bord. Ce n'est pas assez, seul, pour expliquer un déséquilibre (la commande est symétrique), mais un défaut d'offset de 0,1 deg du même ordre ne serait pas détectable sur les angles affichés.

### 5.3 IMU

- Le champ `Accelerometer_orientation` du log vaut l'identité à tout instant : inutilisable. Les valeurs ci-dessous viennent de l'accélération brute (`Accelerometer_linearAcceleration`).
- Le signal est **remis à zéro au démarrage de l'essai** (posture 1 : roulis et tangage = 0,000 deg exactement). On ne connaît donc que des variations, pas l'inclinaison absolue.

| État | Roulis (deg) | Tangage (deg) |
|---|---|---|
| Posture 1 (sol) | 0,001 | 0,00 |
| Posture 2 (sol) | 0,040 | 12,56 |
| Posture 3 (sol) | -0,113 | 20,34 |
| En l'air (t = 940-3600 s) | -0,42 | -3,96 |

- Entre postures, le roulis du bassin varie de 0,04 et -0,11 deg alors que l'écart de charge passe de 100 N à 20 N. Le bruit de l'IMU (environ 0,03 m/s^2 par échantillon, soit 0,16 deg, moyenné sur des centaines d'échantillons) est du même ordre que l'effet cherché (0,1 deg). **L'IMU ne permet pas de conclure sur l'inclinaison du bassin.**

### 5.4 Pourquoi l'écart est le plus grand jambes tendues (hypothèse)

- La différence de force entre les pieds est de l'ordre de `raideur verticale de la jambe x différence de hauteur effective`.
- Une jambe presque tendue se comporte comme une barre : raideur verticale très grande. Une jambe pliée est un ressort : raideur plus faible. Un défaut de 0,2 mm (un offset d'articulation de 0,1 deg y suffit) peut donc donner une centaine de newtons jambes tendues, et beaucoup moins genoux pliés.
- Ordre de grandeur : la compliance verticale croît comme le carré de la distance entre le genou et la ligne hanche-cheville, soit `sin^2(flexion / 2)` pour deux segments égaux. Entre 35,6 et 71,4 deg de genou, le rapport est d'environ 3,6 ; l'écart de charge a été divisé par 5 (100 N -> 20 N). Même ordre de grandeur, **sans que ce soit une preuve**.
- Les efforts horizontaux opposés sous les pieds (Fx, Fy de +-10 à 15 N) qui disparaissent avec la flexion vont dans le même sens (jambes moins précontraintes).

### 5.5 Origine possible : une hauteur effective différente sous chaque pied

Hypothèse : les deux jambes n'ont pas exactement la même **hauteur effective**, sans que les angles affichés ne le montrent. Sources possibles : offset de zéro d'une articulation (0,1 deg ~ 0,2 mm), semelle ou patin plus mince d'un côté (usure), sol non plat. Comme pour une table dont un pied est plus long : le pied le plus long porte le plus de force, et le corps penche légèrement du côté du pied le plus court. Ici le pied droit porte plus : le droit serait effectivement plus long, ou le gauche plus court. Un roulis de bassin de 0,1 deg correspond à environ 0,15 mm de différence de hauteur entre les pieds (espacement 15 cm), soit plusieurs dizaines de newtons avec des jambes tendues.

**Rien de cela n'est mesuré** : c'est la logique de la table, pas une preuve. Si la semelle est usée sur un bord seulement, le pied devient un coin et c'est surtout le point d'appui (`Mx`) qui change.

## 6. Courants moteur et gains des chevilles

### 6.1 Courants (`tauIn`, ampères)

| Articulation | Posture 1 G / D | Posture 2 G / D | Posture 3 G / D |
|---|---|---|---|
| Genou | -0,426 / -0,807 | -0,662 / -0,824 | -1,094 / -1,173 |
| Cheville P | -1,116 / 1,477 | 0,038 / 2,271 | 2,184 / 3,929 |
| Hanche P | 0,276 / 0,322 | 0,021 / 0,140 | 0,556 / 0,650 |

Rapport droite / gauche du genou : 1,9 / 1,24 / 1,07. Rapport de charge : 1,48 / 1,11 / 1,08. Le courant du genou suit donc l'asymétrie de charge. Les courants sont faibles (quelques dixièmes d'ampère) : la charge est probablement portée surtout par la mécanique (hypothèse non vérifiée), donc leur valeur absolue dit peu.

### 6.2 `KI[2]` des chevilles

Lecture, écriture et sauvegarde (commandes `sdo`) :

```
sdo LAnkleRPI float kp 3                    # lecture, ne jamais ajouter d'argument
sdo LAnkleRPI float ki 2
sdo LAnkleRPI int gs 2
sdo LAnkleRPI float ki 2 2.0                # écriture
sdo LAnkleRPI uint 0x1010 1 1702257011      # sauvegarde (0x65766173 = "save")
```

État avant : `KP[3]=60`, `KI[2]=0.02`, `GS[2]=0` sur les 4 drivers. Après : `KP[3]=60`, `KI[2]=2.0`, `GS[2]=0`, relu après sauvegarde. Les noms de drivers `LAnkleRPI`, `LAnkleRPO`, `RAnkleRPI`, `RAnkleRPO` sont les seuls drivers cheville de `RHPS1.conf` ; leur correspondance avec Drive09/10/14/15 n'a pas été vérifiée.

**Incident.** Pendant une vérification, une boucle a ajouté un `0` aux commandes de lecture de `LAnkleRPI` : `KP[3]` et `KI[2]` ont été écrits à 0 pendant quelques secondes (servos éteints, aucun contrôleur). Restaurés à 60 et 0.02, puis sauvegardés à nouveau sur les 4 drivers. Règle : **un argument en trop transforme une lecture en écriture**.

## 7. Dérive

**Sous charge** (posture 2, 240 s immobile) : droit 276,1 -> 275,1 N, gauche 248,0 -> 247,5 N, soit moins de 0,4 %.

**Zéro en l'air** (robot soulevé de t = 880 s à la fin du log) :

| t (s) | RF Fz (N) | LF Fz (N) |
|---|---|---|
| 100-120 | -2,1 | -1,2 |
| 560-600 | -2,7 | +0,4 |
| 940-1000 | -2,7 | +3,8 |
| 1200-1320 | -1,8 | +4,8 |
| 2040-2160 | 0,0 | +6,5 |
| 3000-3120 | +0,1 | +7,5 |
| 3480-3600 | -1,1 | +7,2 |

Le zéro gauche monte d'environ 9 N (3,5 % de la charge) en ~50 min, vite pendant les 15 premières minutes puis il se stabilise. Le droit varie de +-3 N. **Recommandation** : refaire un zéro robot en l'air, capteurs chauds, juste avant chaque essai.

**Écart de somme.** La somme des forces vaut 514 à 536 N (52,4 à 54,6 kg) alors que le robot pèse 58,24 kg (571 N). Si le zéro est fait pieds suspendus, la somme devrait valoir tout le poids. Il manque donc 6 à 10 %. Causes possibles (non départagées) : une partie du poids reprise par un câble ou le portique, une masse de 58,24 kg qui compte des éléments non posés, une échelle droite légèrement basse (pente 0,98).

## 8. À faire pour conclure

1. **Lire `CountPerForce`** des 4 capteurs au démarrage d'`io_main` (lignes `[adresse] CountPerForce = ...` sur la console). Si le gauche diffère du droit, ou vaut 1000, c'est la cause : corriger à la source. Sinon, vérifier la calibration active du capteur et son certificat ATI (les indices SDO de calibration ne sont pas connus, ne pas les deviner).
2. **Refaire la balance sur le pied gauche avec le gain actif** (3 à 4 charges) : le rapport doit passer de 1,12 à ~1,0. Contrôler aussi les moments (poids à bras de levier connu) avant d'appliquer un gain sur les moments.
3. **Peser chaque pied en posture task** (balance sous un pied, cale de même épaisseur sous l'autre, puis inverser) pour comparer le partage réel (balance) au partage des capteurs. Cela tranche « robot déséquilibré » contre « capteur ».
4. **Test de cale à deux angles de genou** (test central pour l'origine) :
   - Posture task, genou 35,6 deg : relever `Fz` D / G et `Mx` D / G (dernière image de `bodystat`, 2 s après lancement, ou moyenne du log sur 10 s).
   - Glisser une cale de 0,5 mm sous le pied gauche, repartir de la même posture, relever à nouveau. Puis cale de 1,0 mm.
   - Refaire à genou 71,4 deg.
   - Lecture : si l'écart de charge bouge de plusieurs dizaines de newtons par demi-millimètre à genou 35,6 deg et beaucoup moins à 71,4 deg, la raideur des jambes tendues amplifie un défaut de hauteur (§5.4), et l'épaisseur de cale qui annule l'écart donne la correction mécanique. Si le partage ne bouge presque pas, la cause est ailleurs (offsets, appui).
   - Précautions : même posture et même placement entre mesures (la reproductibilité est d'environ 10 N), cale rigide, robot posé sans choc.
5. **Test du roulis de cheville** : changer l'angle de roulis de cheville de +-0,1 deg, symétriquement puis d'un seul côté, par petits pas, et relire `Fz` et `Mx`. Il montre si la chaîne de roulis (+-0,107 deg) pilote l'appui. Mouvement réel du robot : un pas à la fois, avec accord.
6. **Balayage de posture** : genou 25, 35, 49, 60, 71 deg, plusieurs placements par posture, pour confirmer que l'écart diminue avec la flexion et mesurer la reproductibilité (~10 N par placement).
7. **Causes mécaniques** : comparer les offsets d'articulations (`offset.dat`, du 2026-07-31) à gauche et à droite, mesurer l'épaisseur des semelles et la planéité du sol.
8. **Contrôleur du collègue** : savoir s'il fait un zéro des forces au démarrage, s'il suppose un partage 50 / 50, et s'il a été réglé avec `KI[2]=0.02` ou 2.0.

## 9. Fichiers

Dans `logs/tmp/` :

- `rhps1_halfsit_2026-11-04_keys_t0-1010.bin.xz` (31 Mo) : log `HalfSitPose` du 2026-11-04 19:42, t = 0..1010 s, clés `t`, forces des deux pieds, `qIn`, `qOut`, `tauIn`, orientation de l'accéléromètre. Les 45 min suivantes sont le robot en l'air (pas gardées en détail).
- `rhps1_halfsit_2026-11-04_summary_1Hz.csv.gz` (133 Ko) : toute la session (3615 s) à 1 Hz, avec forces, commandes de genou, cheville et buste, courants.
- `rhps1_halfsit_2026-11-04_imu_accel_1Hz.csv.gz` : accélération brute de l'IMU (`ax, ay, az`, m/s^2) à 1 Hz sur toute la session. Le champ `Accelerometer_orientation` du fichier `keys` vaut l'identité et n'est pas exploitable ; le roulis et le tangage du §5.3 viennent de ce fichier (`roulis = atan2(ay, az)`, `tangage = atan2(ax, hypot(ay, az))`, signal remis à zéro au démarrage de l'essai).
- `rhps1_halfsit_2026-11-04_hrpsys.log.gz` : sortie de la session `io_main` / rtcd.

Dans `docs/` : `hrp5p-iob_force_gain.patch` (le gain IOB), ce rapport et le résumé en anglais.

Réextraire les clés utiles d'un log complet, puis exporter en CSV :

```
mc_bin_utils extract --in /tmp/mc-control-HalfSitPose-<date>.bin --out hs \
  --keys t LeftFootForceSensor RightFootForceSensor qOut qIn tauIn Accelerometer_orientation
mc_bin_utils convert --in hs.bin --out hs.csv
```

Indices des articulations dans `qIn` / `qOut` / `tauIn` (même ordre que `bodystat`) : 2 `L_CROTCH_P`, 3 `L_KNEE_P`, 5 `L_ANKLE_P`, 7 `CHEST_P`, 10 `R_CROTCH_P`, 11 `R_KNEE_P`, 13 `R_ANKLE_P`. Les angles du log sont en radians.

## 10. Limites de ce rapport

- Une seule mesure stable par posture, et les placements du robot varient d'environ 10 N.
- Les mesures à la balance viennent de l'utilisateur ; elles sont cohérentes entre elles mais n'ont pas été refaites après la correction.
- La somme des forces est 6 à 10 % sous le poids du robot, sans explication confirmée.
- Le gain du pied gauche (0,89) est un contournement fondé sur la balance, pas une correction de la cause.
- Le lien entre ces constats et la chute à droite du contrôleur du collègue n'est pas démontré.
- L'ajustement du §5.1 (centre de masse ~4 mm, espacement ~15 cm) repose sur 3 points d'un seul run et sur l'hypothèse que `Mx` est le moment de roulis ; la mesure d'espacement des pieds n'a pas été vérifiée.
- L'IMU du log est remise à zéro au démarrage de l'essai et son bruit est de l'ordre de l'effet cherché ; elle ne donne pas l'inclinaison absolue du bassin.
- L'explication par la raideur des jambes (§5.4) et par la hauteur effective (§5.5) est une hypothèse cohérente avec la tendance, pas une mesure.
