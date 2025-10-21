library(dplyr)
library(readr)
library(GenomicRanges)
library(rtracklayer)

manifest_path <- "C:\\Users\\omnia\\OneDrive - University of Jeddah\\PhD progress\\DNA_methyalation\\EPIC.hg38.manifest.tsv"
gtf_path <- "C:\\Users\\omnia\\OneDrive\\Desktop\\Homo_sapiens.GRCh38.109.chr.gtf"


manifest <- read.delim(manifest_path, header = TRUE, sep = "\t", stringsAsFactors = FALSE)
head(manifest)


# Keep only rows with complete coordinates
manifest_clean <- manifest %>%
  filter(!is.na(CpG_chrm), !is.na(CpG_beg), !is.na(CpG_end))

cat("Removed", nrow(manifest) - nrow(manifest_clean), "probes with missing coordinates\n")


# Load gene annotation from GTF
gtf <- import(gtf_path)
genes <- gtf[gtf$type == "gene"]

# removes the “chr” prefix (so "chr1" becomes "1") because our manifest has chromosome names like "chr1", "chr2", andd The GTF file from Ensembl uses names without "chr" — e.g. "1", "2", "X", etc.
# So when R tries to find overlaps (findOverlaps()), it says “no sequence levels in common”, because "chr1" ≠ "1".
manifest_clean$CpG_chrm <- gsub("^chr", "", manifest_clean$CpG_chrm)

gr_cpg <- GRanges(
  seqnames = manifest_clean$CpG_chrm,
  ranges = IRanges(start = manifest_clean$CpG_beg, end = manifest_clean$CpG_end),
  Probe_ID = manifest_clean$Probe_ID
)

# sanity check
seqlevels(gtf)[1:10]

# Keep only standard chromosomes
canonical_chr <- as.character(c(1:22, "X", "Y"))
gr_cpg <- gr_cpg[as.character(seqnames(gr_cpg)) %in% canonical_chr]
genes  <- genes[as.character(seqnames(genes)) %in% canonical_chr]


hits <- findOverlaps(gr_cpg, genes)


mapped <- data.frame(
  Probe_ID = gr_cpg$Probe_ID[queryHits(hits)],
  Gene_ID = genes$gene_id[subjectHits(hits)],
  Gene_Name = genes$gene_name[subjectHits(hits)],
  Gene_Type = genes$gene_biotype[subjectHits(hits)]
)

write.csv(mapped,
          "C:/Users/omnia/OneDrive - University of Jeddah/PhD progress/DNA_methyalation/CpG_to_Gene_hg38.csv",
          row.names = FALSE)

