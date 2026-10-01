# One streaming pass over a processed beta matrix.
#   - writes the rows named in KEEP verbatim to KEEPOUT
#   - accumulates per-sample means over the OPENSEA and ISLAND probe classes
# Memory is bounded by the three probe lists; the matrix itself is never held.
BEGIN {
  FS = OFS = "\t"
  while ((getline p < KEEP)    > 0) if (p != "") keep[p]  = 1
  while ((getline p < OPENSEA) > 0) if (p != "") osea[p]  = 1
  while ((getline p < ISLAND)  > 0) if (p != "") isl[p]   = 1
}
NR == 1 { ncol = NF; header = $0; print > KEEPOUT; next }
{
  p = $1
  if (p in keep) print > KEEPOUT
  if (p in osea) { for (j = 2; j <= NF; j++) if ($j+0 == $j && $j != "") { so[j] += $j; co[j]++ } ; no++ }
  else if (p in isl) { for (j = 2; j <= NF; j++) if ($j+0 == $j && $j != "") { si[j] += $j; ci[j]++ } ; ni++ }
  nrow++
}
END {
  split(header, h, FS)
  print "sample_id", "n_opensea", "mean_opensea", "n_island", "mean_island" > MEANOUT
  for (j = 2; j <= ncol; j++)
    print h[j], co[j], (co[j] ? so[j]/co[j] : "NA"), ci[j], (ci[j] ? si[j]/ci[j] : "NA") > MEANOUT
  print "rows_read=" nrow "  opensea_rows=" no "  island_rows=" ni "  cols=" ncol > "/dev/stderr"
}
