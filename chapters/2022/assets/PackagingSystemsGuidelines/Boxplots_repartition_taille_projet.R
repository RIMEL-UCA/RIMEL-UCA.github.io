dataProjetInitial = read.csv(file.choose(), header = TRUE, sep = ";")

summary(dataProjetInitial$Taille_ko)
boxplot(dataProjetInitial$Taille_ko, horizontal = TRUE, main="Répartition des projets en fonction de leur taille", xlab="Taille des projets en kilo octet")
####################-Suppression des valeurs aberrantes-####################

# 1,5 x ecart interquartile (q3 - q1)
q1 = 1518
q3 = 29612

interQ = q3 - q1
seuil = q3 + 1.5 * interQ 

dataProjetNew = dataProjetInitial[dataProjetInitial$Taille_ko < seuil,]

summary(dataProjetNew$Taille_ko)
boxplot(dataProjetNew$Taille_ko, horizontal = TRUE, main="Répartition des projets en fonction de leur taille", xlab="Taille des projets en kilo octet")

