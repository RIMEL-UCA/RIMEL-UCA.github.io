library(VennDiagram)
grid.newpage()

#View(matrice)
area1 = 0.55
area2 = 6
area3 = 0.73
area4 = 30.18
n12 = 0.18
n13 = 8.36
n14 = 2.73
n23 = 0
n24 = 2.91
n34 = 2
n123 = 1.82
n134 = 6.73
n234 = 0.18
n124 = 0.73
n1234 = 2.73
none = 0


total <- area1 + area2 + area3 + area4 + n12 + n13 + n14 + n23 + n24 + n34 + n123 + n234 + n124 + n134 + n1234 + none
print(total)

draw.quad.venn(
  direct.area = TRUE,
  area.vector = c(area3,n34,area4,n13,n134,n1234,n234,n24,area1,n14,n124,n123,n23,area2,n12),
  category = c("SPRING", "ORM", "DOCKER", "CI"),
  fill = c("dodgerblue", "goldenrod1", "darkorange1", "seagreen3"),
  cat.col = c("dodgerblue", "goldenrod1", "darkorange1", "seagreen3"),
  cat.cex = 2,
  margin = 0.05,
  ind = TRUE,
  filename = '#14_venn_diagramm.png',
  output=TRUE
)


# draw.quad.venn(
#   area1 = area1,
#   area2 = area2,
#   area3 = area3,
#   area4 = area4,
#   n12 = n12,
#   n13 = n13,
#   n14 = n14,
#   n23 = n23,
#   n24 = n24,
#   n34 = n34,
#   n123 = n123,
#   n124 = n124,
#   n134 = n134,
#   n234 = n234,
#   n1234 = n1234,
#   category = c("GIT", "GIT_ADVANCED", "JDIME_SEMI", "JDIME_STRUCT"),
#   fill = c("dodgerblue", "goldenrod1", "darkorange1", "seagreen3"),
#   cat.col = c("dodgerblue", "goldenrod1", "darkorange1", "seagreen3"),
#   cat.cex = 2,
#   margin = 0.05,
#   ind = TRUE
# )
