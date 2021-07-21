# Body Segmentation durch Semantic Segmentation

- Olcay Gören

## Problembeschreibung
Ziel dieses Projektes war es Menschen in Bildern zu lokalisieren, mittels Semantic Segmentation. Anschließend sollte eine Foreground Background segmentation durchgeführt werden. Ähnlich wie es in gänigen Foto-Apps Heutzutage gemacht wird. Jedoch bin ich dazu im Rahmen dieses Projektes nicht mehr gekommen.


## Datensatz
Für dieses Projekt wurde der Datensatz [Segmentation Full Body MADS Dataset](https://www.kaggle.com/tapakah68/segmentation-full-body-mads-dataset)  benutzt. Dieser enthält 1192 Bilder und die zugehörigen Masken. In jedem Bild ist eine einzelne Personen zu sehen, die eine sportliche Tätigkeit ausführt. Der Datensatz wurde aufgeteilt in:
- 80% Trainingsbilder
- 10% Validbilder
- 10% Testbilder

## Architektur
Um dieses Projekt zu bewerkstelligen wurde auf das Semantic Segmentation Model DeepLabv3+ gegriffen.
Ich habe mich für das Model entschieden, da es [89%  Genauigkeit im Pascal-Voc-2012 ](https://paperswithcode.com/sota/semantic-segmentation-on-pascal-voc-2012) erzielt hat und open-sourced ist. Da Pytorch von Haus aus kein DeepLabv3+ unterstützt, habe ich die Bibliothek [segmentation_models](https://github.com/qubvel/segmentation_models.pytorch) verwendet. Im Paper von DeepLabv3+ wurden die besten Ergebnisse mit dem xception backbone erzielt, dieses konnte ich nicht verwenden da mein Rechner an seine Grenzen gestoßen ist, weswegen ich mich für das lightweight backbone mobilenet_v2 entschieden habe, welches pre-trained auf dem imagenet Datensatz ist. 

## Trainingsparameter
- Loss Function = Dice
- Optimizer = Adam
- training batch_size = 15
- valid batch_size = 4
- epoch = 8
- input image_size = 512 x 384


## Training
Beim Training habe ich auf **transfer learning** gesetzt. Dabei habe ich die Variante benutzt, wo das **gesamte Model** von 0 auf trainiert wird.
Das Training hat insgesamt 6:30 Stunden gedauert und hat meinen Rechner an seine Grenzen gebracht. Durch das Training konnte eine **Genauigkeit von 90%** erzielt werden. 
Hier die Ergebnisse: 
![](https://github.com/OlcayGoeren/cv_project/blob/master/train_results/res1.png?raw=true)
![enter image description here](https://github.com/OlcayGoeren/cv_project/blob/master/train_results/res2.png?raw=true)
![enter image description here](https://github.com/OlcayGoeren/cv_project/blob/master/train_results/res3.png?raw=true)
Warum meine Vorhergesehenen Bilder so verschwommen sind, kann ich mir nicht wirklich erklären. Möglicher Grund hierfür könnte die geringe Epoch Anzahl sein.
Ich vermute, dass die blauen Stühle mit in die vorhergesehenen Masken aufgenommen wurden, weil in vielen Bildern die Personen mit blauem Licht angestrahlt werden.

## Fazit
Mittels des DeepLabv3+ Models konnten Pixelgruppen zu Personen zugeordnet werden, dabei konnte eine Genauigkeit von 90% ermitteln werden. Damit das training weniger Resourcen verbraucht, würde ich bei meinem nächsten Model, mehrere Zwischenevaluierungsschritte einbauen:
-  Auf die Auswahl des Backbones achten. Ich würde mit dem kleinstmöglichem Backbone anfangen und mich Schrittweise an größere Backbones antasten. Dabei würde ich die einzelnen Models stetig miteinander vergleichen. 
- Bilder, von Anfang an in kleinere Größen resizen, aber auch hier schauen ob zu kleine Bilder nicht die Accurancy mindern
- Pre-Trained Models nicht von vorne trainieren. Schauen ob man mit dem alternativen Transfer Learning Ansatz, wo ein Teil der Layer eingefroren werden, zu besseren Ergebnissen kommen kann.

  

