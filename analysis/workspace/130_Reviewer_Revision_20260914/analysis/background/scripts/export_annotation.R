suppressPackageStartupMessages(library(IlluminaHumanMethylation450kanno.ilmn12.hg19))
a <- getAnnotation(IlluminaHumanMethylation450kanno.ilmn12.hg19)
keep <- c('Name','chr','pos','Type','Islands_Name','Relation_to_Island','UCSC_RefGene_Name','UCSC_RefGene_Group')
out <- commandArgs(trailingOnly=TRUE)[1]
write.table(as.data.frame(a[,keep]),out,sep='\t',quote=FALSE,row.names=FALSE)
cat('IlluminaHumanMethylation450kanno.ilmn12.hg19',as.character(packageVersion('IlluminaHumanMethylation450kanno.ilmn12.hg19')),'\n')
