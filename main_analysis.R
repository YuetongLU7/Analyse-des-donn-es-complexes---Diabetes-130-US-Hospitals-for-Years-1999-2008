# Project: Diabetes Hospital Dataset Analysis
# Authors: Rim El Fatihi & Yuetong Lu & Guilherme Peres Tatagiba

rm(list=ls())
#setwd("C:/Users/guipe/Modelisation statistique/Analyse-des-donn-es-complexes---Diabetes-130-US-Hospitals-for-Years-1999-2008")
#getwd()

# Chargement des bibliothèques
library(dplyr)
library(ggplot2)
library(VIM)         # kNN
library(mice)        # mice (EM)
library(forecast)    # Forecasting
library(caret)       # Préprocessing

# PARTIE 1 (TRAITEMENT DES VALEURS MANQUANTES)
#------------------------------------------------------------------------------------

# 1) Chargement des données
diabetes <- read.csv("dataset/diabetic_data.csv", na.strings = c("", "NA", "?"))
cat("Dimensions initiales :", dim(diabetes), "\n")
gc()

# 2) Prétraitement initial
cat("\n== Prétraitement initial ==\n")

# 2.1 Supprimer 'weight' (trop de NA)
if("weight" %in% names(diabetes)) {
  diabetes$weight <- NULL
  cat("-> colonne 'weight' supprimée\n")
}

# 2.2 Convertir chaînes en facteurs et nettoyer entrées vides
for(v in names(diabetes)) {
  if(is.character(diabetes[[v]])) {
    diabetes[[v]] <- trimws(diabetes[[v]])
    diabetes[[v]][diabetes[[v]] == ""] <- NA
    diabetes[[v]] <- as.factor(diabetes[[v]])
  }
}

# 2.3 Remplacer NA massifs par catégories explicites (utile pour kNN/mice)
if("payer_code" %in% names(diabetes)) {
  diabetes$payer_code <- addNA(as.factor(diabetes$payer_code))
  levels(diabetes$payer_code)[is.na(levels(diabetes$payer_code))] <- "Unknown_payer"
  cat("-> payer_code : NA => 'Unknown_payer'\n")
}
if("medical_specialty" %in% names(diabetes)) {
  diabetes$medical_specialty <- addNA(as.factor(diabetes$medical_specialty))
  levels(diabetes$medical_specialty)[is.na(levels(diabetes$medical_specialty))] <- "Unknown_specialty"
  cat("-> medical_specialty : NA => 'Unknown_specialty'\n")
}

gc()

# 3) Regroupement des codes diag_1/diag_2/diag_3 (ICD-9) en macro-catégories
cat("\n== Regroupement ICD-9 pour diag_1/2/3 ==\n")
group_diag_icd9 <- function(code) {
  if(is.na(code) || code == "") return(NA)
  s <- as.character(code)
  s <- trimws(s)
  # Codes commençant par 'V' ou 'E'
  if(grepl("^[Vv]", s)) return("V_codes")
  if(grepl("^[Ee]", s)) return("E_codes")
  # essayer extraire partie numérique (premiers 3 chars)
  num <- suppressWarnings(as.numeric(substr(s, 1, 3)))
  if(is.na(num)) return(NA)
  if (num >= 1   & num <= 139) return("infectious")
  if (num >= 140 & num <= 239) return("neoplasms")
  if (num >= 240 & num <= 279) return("endocrine")
  if (num >= 280 & num <= 289) return("blood")
  if (num >= 290 & num <= 319) return("mental")
  if (num >= 320 & num <= 389) return("nervous")
  if (num >= 390 & num <= 459) return("circulatory")
  if (num >= 460 & num <= 519) return("respiratory")
  if (num >= 520 & num <= 579) return("digestive")
  if (num >= 580 & num <= 629) return("genitourinary")
  if (num >= 630 & num <= 679) return("pregnancy")
  if (num >= 680 & num <= 709) return("skin")
  if (num >= 710 & num <= 739) return("musculoskeletal")
  if (num >= 740 & num <= 759) return("congenital")
  if (num >= 760 & num <= 779) return("perinatal")
  if (num >= 780 & num <= 799) return("symptoms")
  if (num >= 800 & num <= 999) return("injury")
  return("other")
}

for(col in c("diag_1","diag_2","diag_3")) {
  if(col %in% names(diabetes)) {
    new_col <- paste0(col, "_group")
    diabetes[[new_col]] <- factor(sapply(as.character(diabetes[[col]]), group_diag_icd9))
    cat("-> créé :", new_col, "avec", length(levels(diabetes[[new_col]])), "niveaux\n")
  }
}
gc()

# 4) Vérifier variables avec NA après preprocessing
missing_summary <- sapply(diabetes, function(x) sum(is.na(x)))
missing_vars <- names(missing_summary[missing_summary > 0])
cat("\nVariables avec NA (après preprocessing) :\n")
print(missing_summary[missing_summary > 0])
gc()

# 5) IMPUTATION 1 : kNN optimisé (variable-par-variable)
cat("\n== IMPUTATION kNN (variable par variable, imp_var=FALSE) ==\n")

diab_knn <- diabetes

#Suppression : 

diab_knn$diag_1 <- NULL
diab_knn$diag_2 <- NULL
diab_knn$diag_3 <- NULL

# Choisir variables numériques robustes pour distance
numeric_candidates <- intersect(names(diab_knn)[sapply(diab_knn, is.numeric)],
                                c("time_in_hospital","num_lab_procedures","num_procedures","num_medications","number_diagnoses"))
if(length(numeric_candidates) < 1) {
  numeric_candidates <- names(diab_knn)[sapply(diab_knn, is.numeric)][1:min(3, sum(sapply(diab_knn, is.numeric)))]
}
dist_vars <- numeric_candidates
cat("Variables utilisées pour la distance kNN :", paste(dist_vars, collapse = ", "), "\n")

# recalculer liste NA car on a créé des diag_group
missing_summary <- sapply(diab_knn, function(x) sum(is.na(x)))
vars_to_impute <- names(missing_summary[missing_summary > 0])
cat("Nombre de variables à imputer (kNN loop):", length(vars_to_impute), "\n")
print(vars_to_impute)

for(var in vars_to_impute) {
  cat("kNN -> imputer :", var, " (NA:", sum(is.na(diab_knn[[var]])), ")\n")
  # sauter si déjà plein
  if(sum(is.na(diab_knn[[var]])) == 0) next
  # préparer subset minimal
  cols_for_kNN <- unique(c(var, dist_vars))
  subset_df <- diab_knn[, cols_for_kNN, drop = FALSE]
  tryCatch({
    out <- kNN(subset_df, variable = var, k = 5, imp_var = FALSE, dist_var = dist_vars, weightDist = FALSE)
    # remplacer colonne entière (s'il s'agit d'un factor, kNN garde le type)
    diab_knn[[var]] <- out[[var]]
    cat("  -> kNN succès\n")
    gc()
  }, error = function(e) {
    cat("  -> kNN erreur sur", var, ":", conditionMessage(e), " -> fallback median/mode\n")
    if(is.numeric(diab_knn[[var]])) {
      diab_knn[[var]][is.na(diab_knn[[var]])] <- median(diab_knn[[var]], na.rm = TRUE)
    } else {
      moda <- names(sort(table(diab_knn[[var]]), decreasing = TRUE))[1]
      diab_knn[[var]][is.na(diab_knn[[var]])] <- moda
      diab_knn[[var]] <- factor(diab_knn[[var]])
    }
    gc()
  })
}
cat("NA restants après kNN :", sum(is.na(diab_knn)), "\n")
saveRDS(diab_knn, "diabetes_after_knn.rds")
gc()

#############################################
# 6) IMPUTATION 2 : EM avec mice (norm + polyreg)
#############################################

# Copier le dataset kNN
diab_em <- diabetes

#Supprimer les anciennes variables diag_* (trop de catégories)
diab_em$diag_1 <- NULL
diab_em$diag_2 <- NULL
diab_em$diag_3 <- NULL

# S'assurer que toutes les variables qualitatives sont des facteurs
for (v in names(diab_em)) {
  if (is.character(diab_em[[v]])) {
    diab_em[[v]] <- factor(diab_em[[v]])
  }
}

# Méthodes automatiques
methods_em <- make.method(diab_em)

# Attribution manuelle : numérique → norm, factor → polyreg
for (v in names(diab_em)) {
  if (any(is.na(diab_em[[v]]))) {
    if (is.numeric(diab_em[[v]])) {
      methods_em[v] <- "norm"
    } else {
      methods_em[v] <- "cart"
    }
  } else {
    methods_em[v] <- ""
  }
}

# Matrice de prédicteurs : tout utiliser sauf ID
predM_em <- make.predictorMatrix(diab_em)
predM_em[, "encounter_id"] <- 0
predM_em["encounter_id", ] <- 0

# Imputation EM
imp_em <- mice(diab_em, method = methods_em, predictorMatrix = predM_em,
               m = 1, maxit = 6, print = TRUE)

diab_em_final <- complete(imp_em)


#############################################
# 7) IMPUTATION 3 : Random Forest (mice, method = "rf")
#############################################

diab_rf <- diabetes

# Supprimer diag_1/2/3 (trop de catégories)
diab_rf$diag_1 <- NULL
diab_rf$diag_2 <- NULL
diab_rf$diag_3 <- NULL

# Conversion des variables qualitatives en facteurs
for (v in names(diab_rf)) {
  if (is.character(diab_rf[[v]])) {
    diab_rf[[v]] <- factor(diab_rf[[v]])
  }
}

# Méthodes RF
methods_rf <- make.method(diab_rf)

for (v in names(diab_rf)) {
  if (any(is.na(diab_rf[[v]]))) {
    methods_rf[v] <- "rf"
  } else {
    methods_rf[v] <- ""
  }
}

# Matrice de prédicteurs
predM_rf <- make.predictorMatrix(diab_rf)
predM_rf[, "encounter_id"] <- 0
predM_rf["encounter_id", ] <- 0

# Imputation Random Forest
imp_rf <- mice(diab_rf, method = methods_rf, predictorMatrix = predM_rf,
               m = 1, maxit = 5, print = TRUE)

diab_rf_final <- complete(imp_rf)


# Vérifier NA résiduels et appliquer ranger si nécessaire (sur les trois versions : knn/em)
res_na_knn <- colSums(is.na(diab_knn))
res_na_em  <- colSums(is.na(diab_em))

cat("NA résiduels (kNN) :\n"); print(res_na_knn[res_na_knn > 0])
cat("NA résiduels (EM)  :\n"); print(res_na_em[res_na_em > 0])


# Imputer résiduels kNN
vars_knn_res <- names(res_na_knn[res_na_knn > 0])
if(length(vars_knn_res) > 0) {
  cat("Imputation ranger pour variables restantes dans kNN...\n")
  pred_set <- intersect(names(diab_knn)[sapply(diab_knn, is.numeric)],
                        c("time_in_hospital","num_lab_procedures","num_procedures","num_medications","number_diagnoses"))
  if(length(pred_set) < 1) pred_set <- names(diab_knn)[sapply(diab_knn, is.numeric)][1:min(3, sum(sapply(diab_knn, is.numeric)))]
  for(v in vars_knn_res) {
    cat(" -> ranger imputer", v, "\n")
    try({
      diab_knn <- impute_with_ranger(diab_knn, target = v, pred_vars = pred_set, sample_train = 15000, ntree = 50)
      cat("    NA restants pour", v, ":", sum(is.na(diab_knn[[v]])), "\n")
    }, silent = TRUE)
    gc()
  }
}

# Imputer résiduels EM
vars_em_res <- names(res_na_em[res_na_em > 0])
if(length(vars_em_res) > 0) {
  cat("Imputation ranger pour variables restantes dans EM...\n")
  pred_set2 <- intersect(names(diab_em)[sapply(diab_em, is.numeric)],
                         c("time_in_hospital","num_lab_procedures","num_procedures","num_medications","number_diagnoses"))
  if(length(pred_set2) < 1) pred_set2 <- names(diab_em)[sapply(diab_em, is.numeric)][1:min(3, sum(sapply(diab_em, is.numeric)))]
  for(v in vars_em_res) {
    cat(" -> ranger imputer", v, "\n")
    try({
      diab_em <- impute_with_ranger(diab_em, target = v, pred_vars = pred_set2, sample_train = 15000, ntree = 50)
      cat("    NA restants pour", v, ":", sum(is.na(diab_em[[v]])), "\n")
    }, silent = TRUE)
    gc()
  }
}

# 9) Fallback final : médiane / mode pour garantir 0 NA
final_fix <- function(df) {
  for(nm in names(df)) {
    if(any(is.na(df[[nm]]))) {
      if(is.numeric(df[[nm]])) {
        df[[nm]][is.na(df[[nm]])] <- median(df[[nm]], na.rm = TRUE)
      } else {
        moda <- names(which.max(table(df[[nm]])))
        df[[nm]][is.na(df[[nm]])] <- moda
        df[[nm]] <- factor(df[[nm]])
      }
    }
  }
  return(df)
}

diab_knn_final <- final_fix(diab_knn)
diab_em_final  <- final_fix(diab_em)
# pour MissForest, on n'a imputé que l'échantillon ; ici on n'applique pas missForest sur tout le dataset (trop lourd)
# tu peux choisir entre diab_knn_final / diab_em_final pour les étapes suivantes.

cat("\nNA finaux (kNN/EM) :", sum(is.na(diab_knn_final)), "/", sum(is.na(diab_em_final)), "\n")

# 10) Sauvegardes
saveRDS(diab_knn_final, "diabetes_imputed_knn_final.rds")
saveRDS(diab_em_final,  "diabetes_imputed_em_final.rds")
write.csv(diab_knn_final, "diabetes_imputed_knn_final.csv", row.names = FALSE)
write.csv(diab_em_final,  "diabetes_imputed_em_final.csv",  row.names = FALSE)
saveRDS(diab_rf_final, "diabetes_missforest_sample_imputed.rds")
write.csv(diab_rf_final, "diabetes_missforest_sample_imputed.csv", row.names = FALSE)

cat("\n== Imputation terminée: fichiers écrits ==\n")
cat("Résumé NA finaux (kNN / EM):", sum(is.na(diab_knn_final)), "/", sum(is.na(diab_em_final)), "\n")


#------------------------------------------------------------------------------------
# PARTIE 2 (Sélection de variables et Régularisation)
#------------------------------------------------------------------------------------

library(glmnet)
library(MASS)
library(FactoMineR)
library(factoextra)
library(mixOmics)
library(leaps)
library(pROC)
library(pls)


# 1) Charger les données imputées
diab_knn <- read.csv("diabetes_imputed_knn_final.csv")
diab_em <- read.csv("diabetes_imputed_em_final.csv")
diab_rf <- read.csv("diabetes_missforest_sample_imputed.csv")

# 2) Variables à supprimer parce qu'elles donnent pas des informations pertinentes
cols_to_remove <- c("encounter_id", "patient_nbr", "examide", "citoglipton")

diab_knn <- diab_knn[, !names(diab_knn) %in% cols_to_remove]

diab_em <- diab_em[, !names(diab_em) %in% cols_to_remove]

diab_rf <- diab_rf[, !names(diab_rf) %in% cols_to_remove]


diab_knn$readmit_binary <- ifelse(diab_knn$readmitted == "<30", 1, 0)
diab_knn$readmitted <- NULL

diab_em$readmit_binary <- ifelse(diab_em$readmitted == "<30", 1, 0)
diab_em$readmitted <- NULL

diab_rf$readmit_binary <- ifelse(diab_rf$readmitted == "<30", 1, 0)
diab_rf$readmitted <- NULL


# FONCTION POUR TESTS UNIVARIÉS + BENJAMINI-HOCHBERG
# -----------------------------------------------------
# Cette fonction effectue des tests univariés sur toutes les variables
# et applique la correction de Benjamini-Hochberg pour contrôler le FDR

tests_BH <- function(df, nom_methode, taille_max = 10000) {
  cat("\n", rep("-", 50), sep = "")
  cat("\nTests pour la méthode:", nom_methode, "\n")
  cat(rep("-", 50), "\n", sep = "")
  
  # Échantillonner si le dataset est trop grand
  if (nrow(df) > taille_max) {
    set.seed(123)
    df <- df[sample(1:nrow(df), taille_max), ]
    cat("Échantillon de", taille_max, "observations utilisé\n")
  }
  
  # Préparer les vecteurs pour les résultats
  variables <- character()
  p_values <- numeric()
  tests_utilises <- character()
  
  # Variable cible
  y <- df$readmit_binary
  df <- df[, !colnames(df) %in% "readmit_binary"]
  
  # Boucle sur chaque variable
  for (nom_var in colnames(df)) {
    x <- df[[nom_var]]
    
    # Vérifier s'il y a des valeurs manquantes
    if (any(is.na(x))) {
      next  # Passer à la variable suivante
    }
    
    # Choix du test en fonction du type de variable
    if (is.numeric(x)) {
      # Test de Wilcoxon pour variables numériques
      test_result <- wilcox.test(x ~ y, exact = FALSE)
      p_val <- test_result$p.value
      test_type <- "Wilcoxon"
    } else {
      # Test du chi2 pour variables catégorielles
      tbl <- table(x, y)
      if (nrow(tbl) > 1 && ncol(tbl) > 1) {
        test_result <- chisq.test(tbl)
        p_val <- test_result$p.value
        test_type <- "Chi2"
      } else {
        next  # Table trop petite
      }
    }
    
    # Stocker les résultats
    variables <- c(variables, nom_var)
    p_values <- c(p_values, p_val)
    tests_utilises <- c(tests_utilises, test_type)
  }
  
  # Appliquer la correction de Benjamini-Hochberg
  p_ajustees <- p.adjust(p_values, method = "BH")
  
  # Créer le dataframe des résultats
  results <- data.frame(
    Variable = variables,
    Test = tests_utilises,
    P_value = p_values,
    P_ajustee = p_ajustees,
    Significatif = p_ajustees < 0.05
  )
  
  # Trier par p-value
  results <- results[order(results$P_value), ]
  
  # Statistiques
  cat("Nombre de variables testées:", nrow(results), "\n")
  cat("Variables significatives (p < 0.05 après correction):", 
      sum(results$Significatif), "\n")
  
  # Afficher les 10 variables les plus significatives
  cat("\nTop 10 des variables les plus significatives:\n")
  print(head(results, 10))
  
  return(results)
}

# 4. APPLIQUER LES TESTS AUX 3 MÉTHODES
# -------------------------------------
cat("\n", rep("=", 60), sep = "")
cat("\nDÉBUT DES TESTS MULTIPLES AVEC CORRECTION FDR\n")
cat(rep("=", 60), "\n", sep = "")

# Effectuer les tests pour chaque méthode
resultats_knn <- tests_BH(diab_knn, "kNN Imputation")
resultats_em <- tests_BH(diab_em, "EM Imputation")
resultats_rf <- tests_BH(diab_rf, "RF Imputation")


# Conclusion: Resultat prouve que la structure du dataset est stable malgré la méthode d’imputation.
# Les trois sont paraille, on choisi un des trois apres

# ------------------------------------------------------------------------------------
# PART 3: DIMENSION REDUCTION (PCA & PLS-DA) - PDF OUTPUT VERSION
# ------------------------------------------------------------------------------------

# Clean up graphics environment
graphics.off() 

diab <- diab_knn
# diab <- diab_em

diab$readmit_binary <- factor(diab$readmit_binary, levels = c(0,1))

# Preprocessing
drugs <- c("metformin","repaglinide","nateglinide","chlorpropamide",
           "glimepiride","acetohexamide","glipizide","glyburide",
           "tolbutamide","pioglitazone","rosiglitazone","acarbose",
           "miglitol","troglitazone","tolazamide", 
          "insulin","glyburide-metformin","glipizide-metformin",
           "glimepiride-pioglitazone","metformin-rosiglitazone","metformin-pioglitazone")

diabetes_sample <- diabetes[sample(nrow(diabetes), 20000), ]   # Échantillonnez 20,000 lignes pour éviter un temps d’exécution excessivement long

results <- lapply(drugs, function(d) {
  if(d %in% names(diabetes_sample)) {
    prop <- prop.table(table(diabetes_sample[[d]]))
    data.frame(drug=d, value=names(prop), prop=as.vector(prop))
  }
})
do.call(rbind, results)

map <- function(x) {
  case_when(
    x == "No" ~ 0,
    x == "Steady" ~ 1,
    x == "Down" ~ 2,
    x == "Up" ~ 3,
    TRUE ~ 0 # default
  )
}
# mutate and across of dplyr
drugs <- intersect(drugs, names(diab))
diab <- diab %>%
  mutate(across(all_of(drugs), map))
# Show p value of each drug
drugs_pvalue <- data.frame(Drug = character(), P_Value = numeric(), stringsAsFactors = FALSE)

for(d in drugs) {
  temp_factor <- as.factor(diab[[d]])
  # table: drug levels × readmission (0/1)
  tbl <- table(temp_factor, diab$readmit_binary)
  
  # chi-square
  if(nrow(tbl) > 1 && ncol(tbl) > 1 && sum(tbl) > 0) {
    # reduce warning
    test <- suppressWarnings(chisq.test(tbl))
    drugs_pvalue <- rbind(drugs_pvalue, data.frame(Drug = d, P_Value = test$p.value))
  } else {
    # If chi-square cannot be computed, it assigns p = 1.0 (no significance)
    drugs_pvalue <- rbind(drugs_pvalue, data.frame(Drug = d, P_Value = 1.0))
  }
}

drugs_pvalue <- drugs_pvalue[order(drugs_pvalue$P_Value), ]
print(head(drugs_pvalue, 20))
# -------------------------------------------------------
# Select significant drugs (P < 0.05)
# -------------------------------------------------------

significant_drugs <- drugs_pvalue[drugs_pvalue$P_Value < 0.05, "Drug"]
insignificant_drugs <- setdiff(drugs, significant_drugs)
print(significant_drugs)

diab <- diab[, !names(diab) %in% insignificant_drugs]
diab <- diab[, !names(diab) %in% insignificant_drugs]

# -------------------------------------------------------
# Lasso
# -------------------------------------------------------
# remove Y
exclude_cols <- c("readmit_binary")
X_lasso <- model.matrix( ~ . - 1, data = diab[, !names(diab) %in% exclude_cols])
Y_lasso <- as.numeric(diab$readmit_binary) - 1
print(dim(X_lasso))

set.seed(123)
cv_lasso <- cv.glmnet(
  x = X_lasso, 
  y = Y_lasso, 
  family = "binomial", # 
  alpha = 1,           # alpha=1 Lasso (alpha=0  Ridge)
  type.measure = "auc", # 
  nfolds = 5           # 5 pile validation
)

pdf("plots/Lasso_Selection_Path.pdf", width = 10, height = 6)
plot(cv_lasso)
title("Lasso Cross-Validation Path (AUC)", line = 2.5)
dev.off()

# Best Lambda
best_lambda <- cv_lasso$lambda.min
cat("Lasso Best Lambda:", best_lambda, "\n")
cat("Lasso Biggest AUC:", max(cv_lasso$cvm), "\n")

lasso_coefs <- coef(cv_lasso, s = best_lambda)
lasso_df <- data.frame(
  Feature = rownames(lasso_coefs),
  Coefficient = as.numeric(lasso_coefs)
)

selected_feats <- lasso_df[lasso_df$Coefficient != 0 & lasso_df$Feature != "(Intercept)", ]

selected_feats$Abs_Coef <- abs(selected_feats$Coefficient)
selected_feats <- selected_feats[order(selected_feats$Abs_Coef, decreasing = TRUE), ]

print(head(selected_feats, 30))
# -------------------------------------------------------
# PCA
# -------------------------------------------------------
sig_knn <- subset(resultats_knn, Significatif == TRUE)$Variable
sig_knn <- intersect(sig_knn, names(diab))

# Identify Variables
vars_num <- intersect(
  sig_knn,
  names(diab)[sapply(diab, is.numeric)]
)

cat("Variables numériques significatives utilisées pour la réduction de dimension :\n")
print(vars_num)

# Sampling
set.seed(123)
n_sample <- min(15000, nrow(diab))
diab_sample <- diab[sample(1:nrow(diab), n_sample), ]

# Prepare X (Numeric only) and Y
X_num <- diab_sample[, vars_num]
y <- diab_sample$readmit_binary

# Remove columns with 0 variance
var_check <- apply(X_num, 2, stats::var, na.rm = TRUE)

X_num <- X_num[, var_check > 0]

pdf("plots/Analysis_Results_Part3.pdf", width = 10, height = 7)

# PCA
acp_result <- PCA(X_num, scale.unit = TRUE, ncp = 10, graph = FALSE)

# Scree Plot
print(fviz_eig(acp_result, geom = c("bar", "line"), addlabels = TRUE, ylim = c(0, 50), main = "PCA - Scree Plot"))

# Variable Plot
print(fviz_pca_var(acp_result, 
                   col.var = "contrib",
                   gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
                   repel = TRUE,
                   select.var = list(contrib = 15),
                   labelsize = 4,
                   title = "PCA - Variable Contribution"))

# Individual Plot
print(fviz_pca_ind(acp_result, 
                   geom.ind = "point", 
                   col.ind = y,
                   palette = c("#00AFBB", "#FC4E07"),
                   addEllipses = TRUE,
                   title = "PCA - Individuals"))

# Contribution Plots
print(fviz_contrib(acp_result, choice = "var", axes = 1, top = 11, title = "Contribution to PC1"))
print(fviz_contrib(acp_result, choice = "var", axes = 2, top = 11, title = "Contribution to PC2"))
print(fviz_contrib(acp_result, choice = "var", axes = 3, top = 11, title = "Contribution to PC3"))
print(fviz_contrib(acp_result, choice = "var", axes = 4, top = 11, title = "Contribution to PC4"))
print(fviz_contrib(acp_result, choice = "var", axes = 5, top = 11, title = "Contribution to PC5"))


# sPCA
actual_vars <- ncol(X_num)
keepX_val <- min(5, actual_vars) 

spca_result <- spca(X_num, ncomp = 5, keepX = rep(keepX_val, 5), scale = TRUE)
plotVar(spca_result, comp = c(1, 2), var.names = TRUE, cex = 4, title = "sPCA - Variables")


# PLS (mixOmics)
y_numeric <- as.numeric(y) - 1
pls_result <- pls(X_num, y_numeric, ncomp = 5, scale = TRUE)
plotVar(pls_result, comp = c(1, 2), var.names = TRUE, cex = 4, title = "PLS - Variables")


# sPLS
spls_result <- spls(X_num, y_numeric, ncomp = 5, keepX = rep(keepX_val, 5), scale = TRUE)
plotVar(spls_result, comp = c(1, 2), var.names = TRUE, cex = 4, title = "sPLS - Variables")


# PLS-DA 
plsda_result <- plsda(X_num, y, ncomp = 5, scale = TRUE)

plotIndiv(plsda_result, comp = c(1, 2), group = y, 
          ind.names = FALSE, ellipse = TRUE, legend = TRUE, title = "PLS-DA - Individuals")

plotVar(plsda_result, comp = c(1, 2), var.names = TRUE, cex = 4, title = "PLS-DA - Variables")

plotLoadings(plsda_result, comp = 1, method = 'mean', 
             contrib = 'max', ndisplay = 11, title = "PLS-DA - Loadings PC1")

# Close the PDF device
dev.off()


# Top variables PCA
var_coord <- acp_result$var$coord[, 1:5]
var_contrib <- acp_result$var$contrib[, 1:5]
contrib_total <- rowSums(var_contrib)
top_vars <- head(order(contrib_total, decreasing = TRUE), 11)

print("=== TOP 11 VARIABLES (PCA) ===")
print(data.frame(
  Variable = rownames(var_coord)[top_vars],
  PC1_contrib = round(var_contrib[top_vars, 1], 2),
  PC2_contrib = round(var_contrib[top_vars, 2], 2),
  PC3_contrib = round(var_contrib[top_vars, 3], 2),
  PC4_contrib = round(var_contrib[top_vars, 4], 2),
  PC5_contrib = round(var_contrib[top_vars, 5], 2),
  Total = round(contrib_total[top_vars], 2)
))

# Recap Table
recap <- data.frame(
  Methode = c("ACP", "sPCA", "PLS", "sPLS", "PLS-DA"),
  Type = c("Unsupervised", "Unsupervised", "Supervised", "Supervised", "Supervised"),
  Selection = c("No", "Yes", "No", "Yes", "No"),
  Variance_PC1 = c(round(acp_result$eig[1, 2], 2), NA, NA, NA, NA)
)
print(recap)

# ------------------------------------------------------------------------------------
# SUMMARY TABLE (RECAP)
# ------------------------------------------------------------------------------------

# 1. Method Comparison:
#    - ACP (PCA): A baseline unsupervised method. It shows that the first principal component (PC1)
#      explains approximately 18.8% of the total variance, driven mainly by 'number_inpatient' and 'num_medications'.
#    - sPCA / sPLS (Sparse Methods): These methods applied automatic variable selection (Selection = Yes).
#      By setting coefficient weights to zero for irrelevant variables, they highlighted the most critical features
#      (e.g., number_inpatient, time_in_hospital) and reduced noise.
#    - PLS-DA (Supervised): This method attempted to discriminate between 'Readmis' and 'Non_readmis'
#      using the target variable Y.

# 2. Key Observations from Plots:
#    - Feature Importance: Across all methods, 'number_inpatient' (history of admission) and 
#      'num_medications' (treatment intensity) consistently appear as the top contributors.
#    - Class Separation: The PLS-DA individual plots showed a heavy overlap between the two classes.
#      This indicates that the numeric variables alone do not provide a clear linear separation boundary.

# 3. Conclusion : It is crucial to include categorical variables

#------------------------------------------------------------------------------------
# PARTIE 4 – MODÉLISATION PRÉDICTIVE
# (Ridge, Lasso, Elastic-Net, GAM, Random Forest, XGBoost)
#------------------------------------------------------------------------------------

library(caret)
library(glmnet)
library(pROC)
library(mgcv)
library(randomForest)
library(xgboost)

set.seed(123)
# 4.0 – Dataset de travail : imputation kNN (déjà nettoyé plus haut)
data_model <- diab_knn

# Cible binaire : Yes = readmission < 30 jours, No = sinon
data_model$readmit_binary <- factor(
  ifelse(data_model$readmit_binary == 1, "Yes", "No"),
  levels = c("No", "Yes")   # "Yes" = classe positive
)

cat("\nRépartition de la cible (No / Yes) :\n")
print(table(data_model$readmit_binary))

# ================================
# 4.1 – Split train/test (70% / 30%)
# ================================
set.seed(123)  # reproductible
trainIndex <- createDataPartition(data_model$readmit_binary, p = 0.7, list = FALSE)

train_df <- data_model[trainIndex, ]
test_df  <- data_model[-trainIndex, ]

# On s'assure que la cible est bien factor dans les deux
train_df$readmit_binary <- factor(train_df$readmit_binary, levels = c("No", "Yes"))
test_df$readmit_binary  <- factor(test_df$readmit_binary,  levels = c("No", "Yes"))

# Séparation prédicteurs / cible
pred_cols <- setdiff(names(train_df), "readmit_binary")

X_train_raw <- train_df[, pred_cols, drop = FALSE]
X_test_raw  <- test_df[,  pred_cols, drop = FALSE]

y_train_fac <- train_df$readmit_binary
y_test_fac  <- test_df$readmit_binary

# ==========================================
# 4.1 bis – Supprimer les prédicteurs constants dans le TRAIN
# ==========================================
is_constant <- sapply(X_train_raw, function(col) {
  length(unique(col[!is.na(col)])) < 2
})

if (any(is_constant)) {
  cat("\nVariables constantes supprimées du modèle :\n")
  print(names(X_train_raw)[is_constant])
  
  X_train_raw <- X_train_raw[, !is_constant, drop = FALSE]
  X_test_raw  <- X_test_raw[,  !is_constant, drop = FALSE]
}

# Harmoniser les types : convertir les character en factors
char_cols <- sapply(X_train_raw, is.character)
if (any(char_cols)) {
  X_train_raw[ , char_cols] <- lapply(X_train_raw[ , char_cols, drop = FALSE], factor)
  X_test_raw[  , char_cols] <- lapply(X_test_raw[  , char_cols, drop = FALSE], factor)
}

# Pour chaque factor, imposer les mêmes niveaux dans test que dans train
for (nm in names(X_train_raw)) {
  if (is.factor(X_train_raw[[nm]])) {
    X_test_raw[[nm]] <- factor(X_test_raw[[nm]], levels = levels(X_train_raw[[nm]]))
  }
}

# ==========================================
# 4.2 – Encodage one-hot avec dummyVars sur TRAIN + TEST
# ==========================================

# On colle train et test prédicteurs
full_X_raw <- rbind(X_train_raw, X_test_raw)
n_train <- nrow(X_train_raw)

# dummyVars sur l'ensemble (sans la cible)
dmy <- dummyVars(~ ., data = full_X_raw, fullRank = TRUE)

full_X <- predict(dmy, newdata = full_X_raw)
full_X <- as.data.frame(full_X)

# On recoupe
x_train <- full_X[1:n_train, , drop = FALSE]
x_test  <- full_X[(n_train + 1):nrow(full_X), , drop = FALSE]

# Cible numérique (Yes = 1, No = 0) pour glmnet / xgboost
y_train_num <- ifelse(y_train_fac == "Yes", 1, 0)
y_test_num  <- ifelse(y_test_fac == "Yes", 1, 0)

# Matrices pour glmnet / xgboost
x_train_mat <- as.matrix(x_train)
x_test_mat  <- as.matrix(x_test)

cat("\nDimensions X_train / X_test :", dim(x_train_mat), "/", dim(x_test_mat), "\n")

# 4.3 – Fonction utilitaire : calcul des métriques (AUC, Accuracy, Recall, F1)
compute_metrics <- function(y_true_fac, y_true_num, prob_pred, positive_label = "Yes", threshold = 0.5) {
  # AUC
  roc_obj <- roc(y_true_num, prob_pred)
  auc_val <- as.numeric(auc(roc_obj))
  
  # Classes prédites
  pred_class <- factor(ifelse(prob_pred > threshold, positive_label, "No"),
                       levels = c("No", positive_label))
  
  cm <- confusionMatrix(pred_class, y_true_fac, positive = positive_label)
  
  acc <- as.numeric(cm$overall["Accuracy"])
  recall <- as.numeric(cm$byClass["Recall"])
  f1 <- as.numeric(cm$byClass["F1"])
  
  return(list(
    AUC = auc_val,
    Accuracy = acc,
    Recall = recall,
    F1 = f1
  ))
}

results_models <- data.frame(
  Model = character(),
  AUC = numeric(),
  Accuracy = numeric(),
  Recall = numeric(),
  F1 = numeric(),
  stringsAsFactors = FALSE
)

#------------------------------------------------------------------------------------
# 4.A – Régression logistique pénalisée (Ridge, Lasso, Elastic-Net)
#------------------------------------------------------------------------------------

# A.1 – Ridge (alpha = 0)
cv_ridge <- cv.glmnet(
  x_train_mat, y_train_num,
  family = "binomial",
  alpha = 0,             # Ridge
  nfolds = 5,
  type.measure = "auc"
)
lambda_ridge <- cv_ridge$lambda.min
cat("\n[RIDGE] Lambda optimal :", lambda_ridge, "\n")

ridge_model <- glmnet(
  x_train_mat, y_train_num,
  family = "binomial",
  alpha = 0,
  lambda = lambda_ridge
)

prob_ridge <- as.numeric(predict(ridge_model, newx = x_test_mat, type = "response"))
met_ridge <- compute_metrics(y_test_fac, y_test_num, prob_ridge)

results_models <- rbind(results_models, data.frame(
  Model = "Ridge",
  AUC = met_ridge$AUC,
  Accuracy = met_ridge$Accuracy,
  Recall = met_ridge$Recall,
  F1 = met_ridge$F1
))

# A.2 – Lasso (alpha = 1)
cv_lasso <- cv.glmnet(
  x_train_mat, y_train_num,
  family = "binomial",
  alpha = 1,             # Lasso
  nfolds = 5,
  type.measure = "auc"
)
lambda_lasso <- cv_lasso$lambda.min
cat("\n[LASSO] Lambda optimal :", lambda_lasso, "\n")

lasso_model <- glmnet(
  x_train_mat, y_train_num,
  family = "binomial",
  alpha = 1,
  lambda = lambda_lasso
)

prob_lasso <- as.numeric(predict(lasso_model, newx = x_test_mat, type = "response"))
met_lasso <- compute_metrics(y_test_fac, y_test_num, prob_lasso)

results_models <- rbind(results_models, data.frame(
  Model = "Lasso",
  AUC = met_lasso$AUC,
  Accuracy = met_lasso$Accuracy,
  Recall = met_lasso$Recall,
  F1 = met_lasso$F1
))

# A.3 – Elastic-Net (alpha entre 0 et 1, ex: 0.5)
cv_en <- cv.glmnet(
  x_train_mat, y_train_num,
  family = "binomial",
  alpha = 0.5,           # Elastic-Net
  nfolds = 5,
  type.measure = "auc"
)
lambda_en <- cv_en$lambda.min
cat("\n[Elastic-Net] Lambda optimal :", lambda_en, "\n")

en_model <- glmnet(
  x_train_mat, y_train_num,
  family = "binomial",
  alpha = 0.5,
  lambda = lambda_en
)

prob_en <- as.numeric(predict(en_model, newx = x_test_mat, type = "response"))
met_en <- compute_metrics(y_test_fac, y_test_num, prob_en)

results_models <- rbind(results_models, data.frame(
  Model = "Elastic-Net",
  AUC = met_en$AUC,
  Accuracy = met_en$Accuracy,
  Recall = met_en$Recall,
  F1 = met_en$F1
))

#------------------------------------------------------------------------------------
# 4.B – Modèle additif généralisé (GAM) – prise en compte d'effets non linéaires
#------------------------------------------------------------------------------------

library(mgcv)  # au cas où ce n'est pas déjà chargé
# ------------------------------------------------------------------------------------
# Sélection des variables pour le modèle GAM
# ------------------------------------------------------------------------------------
# Le modèle additif généralisé (GAM) permet de capturer des relations non linéaires
# entre les prédicteurs et la probabilité de réadmission grâce aux fonctions de lissage
# s(x1), s(x2), ..., s(xp).
#
# Cependant, un GAM n'est pas adapté à toutes les variables :
#   - Les variables catégorielles à nombreux niveaux ne peuvent pas être lissées correctement.
#   - Les variables one-hot issues de l'encodage créent des fonctions s(0/1) inutiles.
#   - Le modèle devient instable et difficile à interpréter si trop de variables sont incluses.
#
# Pour cette raison, nous ne retenons PAS toutes les variables significatives des modèles
# précédents (logistique, LASSO, Random Forest). Nous sélectionnons uniquement celles
# qui répondent aux critères suivants :
#
#   1. Variables NUMÉRIQUES et CONTINUES
#   2. Identifiées comme IMPORTANTES dans les modèles précédents
#   3. Présentant une VARIABILITÉ suffisante (pas de colonnes constantes)
#   4. Ayant un sens clinique clair et une relation potentiellement NON LINÉAIRE
#
# Les variables retenues ici (time_in_hospital, num_medications, number_inpatient,
# number_emergency, number_diagnoses) satisfont toutes ces conditions.
#
# Elles permettent au GAM de modéliser des effets lissés interprétables, comme :
#   - augmentation du risque avec le nombre de diagnostics,
#   - effets non linéaires du temps d'hospitalisation,
#   - relation complexe avec le nombre d’inpatient/emergency visits.
#
# Ce choix garantit un modèle GAM :
#   - stable,
#   - interprétable,
#   - utile pour visualiser la forme des relations non linéaires,
#   - complémentaire aux modèles supervisés classiques.
# ------------------------------------------------------------------------------------

# On choisit quelques variables numériques importantes identifiées auparavant
vars_gam <- c("time_in_hospital", "num_medications", "number_inpatient",
              "number_emergency", "number_diagnoses")

# On croise avec les colonnes effectivement présentes dans train_df
vars_gam <- intersect(vars_gam, colnames(train_df))

cat("\nVariables utilisées dans le GAM :\n")
print(vars_gam)

if (length(vars_gam) == 0) {
  stop("Aucune des variables spécifiées pour le GAM n'est présente dans train_df.")
}

# Formule GAM : readmit_binary ~ s(var1) + s(var2) + ...
gam_formula <- as.formula(
  paste("readmit_binary ~",
        paste(paste0("s(", vars_gam, ")"), collapse = " + "))
)

cat("\nFormule GAM utilisée :\n")
print(gam_formula)

# Ajustement du modèle sur le train ORIGINAL (pas les x_train_mat)
gam_model <- gam(gam_formula, data = train_df, family = binomial(link = "logit"))

# Prédictions de probas sur le test_df (même structure que train_df)
prob_gam <- as.numeric(predict(gam_model, newdata = test_df, type = "response"))

# Évaluation (on utilise y_test_fac / y_test_num définis plus haut)
met_gam <- compute_metrics(y_test_fac, y_test_num, prob_gam)

results_models <- rbind(
  results_models,
  data.frame(
    Model = "GAM",
    AUC = met_gam$AUC,
    Accuracy = met_gam$Accuracy,
    Recall = met_gam$Recall,
    F1 = met_gam$F1
  )
)

#------------------------------------------------------------------------------------
# 4.C – Random Forest
#------------------------------------------------------------------------------------

rf_model <- randomForest(
  x = x_train,
  y = y_train_fac,
  ntree = 300,
  mtry = floor(sqrt(ncol(x_train))),
  importance = TRUE
)

prob_rf <- as.numeric(predict(rf_model, newdata = x_test, type = "prob")[, "Yes"])
met_rf <- compute_metrics(y_test_fac, y_test_num, prob_rf)

results_models <- rbind(results_models, data.frame(
  Model = "Random Forest",
  AUC = met_rf$AUC,
  Accuracy = met_rf$Accuracy,
  Recall = met_rf$Recall,
  F1 = met_rf$F1
))

#------------------------------------------------------------------------------------
# 4.D – XGBoost
#------------------------------------------------------------------------------------

dtrain <- xgb.DMatrix(data = x_train_mat, label = y_train_num)
dtest  <- xgb.DMatrix(data = x_test_mat,  label = y_test_num)

params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = 4,
  eta = 0.1,
  subsample = 0.8,
  colsample_bytree = 0.8
)

xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,
  verbose = 0
)

prob_xgb <- as.numeric(predict(xgb_model, newdata = dtest))
met_xgb <- compute_metrics(y_test_fac, y_test_num, prob_xgb)

results_models <- rbind(results_models, data.frame(
  Model = "XGBoost",
  AUC = met_xgb$AUC,
  Accuracy = met_xgb$Accuracy,
  Recall = met_xgb$Recall,
  F1 = met_xgb$F1
))

#------------------------------------------------------------------------------------
# 4.E – Récapitulatif des performances
#------------------------------------------------------------------------------------

cat("\n=== RÉCAPITULATIF DES MODÈLES (AUC, Accuracy, Recall, F1) ===\n")
print(results_models)
