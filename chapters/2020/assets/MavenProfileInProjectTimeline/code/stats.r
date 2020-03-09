#! /usr/bin/env Rscript
d<-scan("stdin", quiet=TRUE)
summary(d)
sd(d)

quantile(d, prob = seq(0,1, length = 11))
