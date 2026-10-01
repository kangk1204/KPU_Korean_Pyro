#!/usr/bin/env Rscript
# Platform availability only; this does not inspect patient methylation.
suppressPackageStartupMessages(library(IlluminaHumanMethylationEPICanno.ilm10b4.hg19))
args <- commandArgs(trailingOnly=TRUE)
root <- if(length(args)) args[[1]] else "120_Korean_External_Validation_20260908"
fixed <- jsonlite::fromJSON(file.path(root, "registry/fixed_probes.json"))
data(list="Locations", package="IlluminaHumanMethylationEPICanno.ilm10b4.hg19")
rows <- do.call(rbind,lapply(names(fixed),function(g) {
  p <- fixed[[g]]
  present <- p %in% rownames(Locations)
  loc <- data.frame(chr=rep(NA_character_,length(p)),pos=rep(NA_real_,length(p)),strand=rep(NA_character_,length(p)))
  if(any(present)) {
    hit <- as.data.frame(Locations[p[present],c("chr","pos","strand"),drop=FALSE])
    loc[present,] <- hit
  }
  data.frame(gene=g,probe=p,epic_annotation_present=present,
    chr=as.character(loc$chr),pos_1based=loc$pos,strand=as.character(loc$strand),
    genome_build="hg19", annotation_package="IlluminaHumanMethylationEPICanno.ilm10b4.hg19",
    annotation_version=as.character(packageVersion("IlluminaHumanMethylationEPICanno.ilm10b4.hg19")),
    patient_matrix_coverage="not_yet_measured")
}))
write.table(rows,file.path(root,"registry/epic_platform_probe_audit.tsv"),
  sep="\t",row.names=FALSE,quote=FALSE,na="")
cat(sum(rows$epic_annotation_present),"/",nrow(rows),"fixed probes in EPIC annotation\n")
