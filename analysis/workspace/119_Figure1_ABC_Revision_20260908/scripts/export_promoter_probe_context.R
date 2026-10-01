#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(IlluminaHumanMethylation450kanno.ilmn12.hg19)
})

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) >= 1) args[[1]] else normalizePath(file.path(dirname(sys.frame(1)$ofile), ".."))
fixed_path <- file.path(root, "registry", "fixed_probes.json")
out_path <- file.path(root, "registry", "reviewer", "probe_context.tsv")

fixed <- jsonlite::fromJSON(fixed_path)
data(list = "Other", package = "IlluminaHumanMethylation450kanno.ilmn12.hg19")
data(list = "Locations", package = "IlluminaHumanMethylation450kanno.ilmn12.hg19")
data(list = "Islands.UCSC", package = "IlluminaHumanMethylation450kanno.ilmn12.hg19")

split_semicolon <- function(x) {
  if (is.na(x) || !nzchar(x)) {
    return(character())
  }
  strsplit(as.character(x), ";", fixed = TRUE)[[1]]
}

rows <- list()
idx <- 1L
for (gene in names(fixed)) {
  for (probe in fixed[[gene]]) {
    rec <- as.data.frame(Other[probe, c("UCSC_RefGene_Name", "UCSC_RefGene_Accession", "UCSC_RefGene_Group")])
    names_v <- split_semicolon(rec$UCSC_RefGene_Name[[1]])
    accessions_v <- split_semicolon(rec$UCSC_RefGene_Accession[[1]])
    groups_v <- split_semicolon(rec$UCSC_RefGene_Group[[1]])
    semicolon_lengths_aligned <- length(names_v) == length(accessions_v) && length(names_v) == length(groups_v)
    if (!semicolon_lengths_aligned) {
      stop(sprintf("Semicolon-aligned RefGene fields differ for %s/%s", gene, probe))
    }
    loc <- as.data.frame(Locations[probe, c("chr", "pos", "strand")])
    island <- as.data.frame(Islands.UCSC[probe, c("Islands_Name", "Relation_to_Island")])
    promoter_idx <- which(names_v == gene & groups_v %in% c("TSS200", "TSS1500"))
    rows[[idx]] <- data.frame(
      fixed_gene = gene,
      probe = probe,
      chr = loc$chr[[1]],
      pos = loc$pos[[1]],
      strand = loc$strand[[1]],
      Islands_Name = island$Islands_Name[[1]],
      Relation_to_Island = island$Relation_to_Island[[1]],
      UCSC_RefGene_Name = rec$UCSC_RefGene_Name[[1]],
      UCSC_RefGene_Accession = rec$UCSC_RefGene_Accession[[1]],
      UCSC_RefGene_Group = rec$UCSC_RefGene_Group[[1]],
      semicolon_lengths_aligned = semicolon_lengths_aligned,
      promoter_match = length(promoter_idx) > 0,
      promoter_groups = paste(unique(groups_v[promoter_idx]), collapse = ";"),
      matched_transcripts = paste(accessions_v[promoter_idx], collapse = ";"),
      matched_pair_indices = paste(promoter_idx, collapse = ";"),
      annotation_source = "IlluminaHumanMethylation450kanno.ilmn12.hg19 v0.6.1 Other",
      stringsAsFactors = FALSE
    )
    idx <- idx + 1L
  }
}

out <- do.call(rbind, rows)
dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
write.table(out, out_path, sep = "\t", quote = FALSE, row.names = FALSE, na = "")
