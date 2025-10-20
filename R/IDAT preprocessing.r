# IDATs processing hg38.R
# Author: Yunchen Xiao
# Edited: Omnia 

# Clear the workspace and load the necessary packages
rm(list = ls())
library(wateRmelon)
# library(bigmelon)
library(rtracklayer)
library(readr)
library(stringr)
library(lumi)

# Need to use the hg38 annotation for EPICv1 array
library(devtools)
#install_github("achilleasNP/IlluminaHumanMethylationEPICanno.ilm10b5.hg38") # Comment this part because we already installed it
library(IlluminaHumanMethylationEPICanno.ilm10b5.hg38)


# Load the .idat files that need to be analyzed in the current batch
dataDirectory <- "/data/DERI-MMH/DNA_meth/IDAT/filtr_p11"   # <--- edit 1 


# Load the .idat files that need to be analyzed in the current batch
gfile.all.batches.analysis <- iadd2(dataDirectory,
                                    gds = "/data/DERI-MMH/DNA_meth/gdc2/P11_850_analysis.gds")  # <--- edit 2 

# Get methylated and unmethylated and betas
mns.all.batches.analysis <- methylated(gfile.all.batches.analysis)[,]
uns.all.batches.analysis <- unmethylated(gfile.all.batches.analysis)[,]
betas.all.batches.analysis <- betas(gfile.all.batches.analysis)

# Remove sex chromosome probes
# Reference to the annotation of EPIC array
annotation.epic.hg38 <- as.data.frame(getAnnotation(IlluminaHumanMethylationEPICanno.ilm10b5.hg38))
ind.sex.mito.probes <- which(annotation.epic.hg38$CHR_hg38== "chrX" | annotation.epic.hg38$CHR_hg38 == "chrY" | annotation.epic.hg38$CHR_hg38 == "chrM")
sex.mito.probes.names <- annotation.epic.hg38$Name[ind.sex.mito.probes]

# Probes in autosomes only
mns.all.batches.analysis.auto <- mns.all.batches.analysis[-which(rownames(mns.all.batches.analysis) %in% sex.mito.probes.names),]
uns.all.batches.analysis.auto <- uns.all.batches.analysis[-which(rownames(uns.all.batches.analysis) %in% sex.mito.probes.names),]
design.type.analysis.auto <- fData(gfile.all.batches.analysis)$DESIGN[-which(row.names(fData(gfile.all.batches.analysis)) %in% sex.mito.probes.names)]

# Verify if all the remaining probes are in autosomes
all.equal(rownames(mns.all.batches.analysis.auto), rownames(uns.all.batches.analysis.auto)) # TRUE
ind.cpg.names.in.hg38.anno <- match(rownames(mns.all.batches.analysis.auto), annotation.epic.hg38$Name)
chr.names <- annotation.epic.hg38$CHR_hg38[ind.cpg.names.in.hg38.anno]
sort(unique(chr.names)) # 22 autosomes only!

# Perform dasen normalisation on autosomes only to avoid
# introducing technical bias from sex chromosomes
print("Dasen normalization on CpGs from autosomes...")
dasen.norm <- adjustedDasen(mns = mns.all.batches.analysis.auto, 
                            uns = uns.all.batches.analysis.auto,
                            onetwo = design.type.analysis.auto,
                            chr = annotation.epic.hg38$chr[match(rownames(mns.all.batches.analysis.auto),
                                                                 annotation.epic.hg38$Name)])


# Now cleaning the data
CpGs.autosomes <- dasen.norm[!rownames(dasen.norm) %in% sex.mito.probes.names, ]

# Read the .tsv file for the masked info of EPIC array in hg38
mask.info.EPIC.hg38 <- read.delim("/data/DERI-MMH/DNA_meth/arrayType/EPIC.hg38.manifest.tsv") # <--- edit 3 No need to edit as we told to use hg38
SNP.associated.CpGs <- mask.info.EPIC.hg38[mask.info.EPIC.hg38$MASK_general == TRUE,]
SNP.associated.CpGs.auto <- SNP.associated.CpGs[SNP.associated.CpGs$probeID %in% rownames(CpGs.autosomes),]
CpGs.cleaned <- CpGs.autosomes[!rownames(CpGs.autosomes) %in% SNP.associated.CpGs.auto$probeID, ]


write.csv(as.data.frame(CpGs.cleaned), "/data/DERI-MMH/DNA_meth/beta-vals/hg38_BetaVal/Part11_850k_Betas.csv",row.names = TRUE) # <--- edit 4  

#write_rds(CpGs.cleaned, "/data/DERI-MMH/DNA_meth/beta-vals/hg38_BetaVal/Part1_850k_Betas.rds")

# Close the opened gfile
closefn.gds(gfile.all.batches.analysis)



############################## Save Beta to M-values ##################################

#path <- "/data/DERI-MMH/DNA_meth/beta-vals/hg38_BetaVal/Part5_850k_Betas.csv"   # <--- edit 5 : Read Beta to convert to M-values
#beta_values <- read.csv(path, header = TRUE, row.names = 1)
#colnames(beta_values) <- str_replace_all(colnames(beta_values), "X", "")
beta_values_mat <- as.matrix(CpGs.cleaned)  # change the dataframe beta values to a mtrix
mvalues <- beta2m(beta_values_mat)   # Convert to M values using the library lumi 
write.csv(as.data.frame(mvalues), "/data/DERI-MMH/DNA_meth/M_vals/Part11_850k_Mval.csv", row.names = TRUE) # <--- edit 5: write M-values to csv

