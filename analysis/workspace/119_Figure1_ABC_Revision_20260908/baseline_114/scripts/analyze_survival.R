#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(survival)
  library(jsonlite)
})

options(digits = 17)

genes <- c("EYA4", "ZNF568", "ZNF793", "SFMBT2", "ADHFE1", "HOXA2", "BEND5", "UNC5C", "RALYL", "GFRA1")

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0 || all(is.na(x))) y else x
}

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_arg) > 0) {
  script_path <- sub("^--file=", "", script_arg[[1]])
  root <- normalizePath(file.path(dirname(script_path), ".."), mustWork = TRUE)
} else {
  root <- normalizePath(getwd(), mustWork = TRUE)
}
if (!dir.exists(file.path(root, "data"))) {
  root <- normalizePath(file.path(getwd(), ".."), mustWork = TRUE)
}

results_dir <- file.path(root, "results")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

clinical_path <- file.path(root, "data", "derived", "clinical.tsv")
methylation_path <- file.path(root, "data", "derived", "methylation_wide.tsv")
clinical <- read.delim(clinical_path, stringsAsFactors = FALSE, check.names = FALSE)
methylation <- read.delim(methylation_path, stringsAsFactors = FALSE, check.names = FALSE)

required_clinical <- c("patient_id", "study_id", "event", "duration_days", "recurrence_primary")
missing_clinical <- setdiff(required_clinical, names(clinical))
if (length(missing_clinical) > 0) {
  stop("clinical.tsv missing required columns: ", paste(missing_clinical, collapse = ", "))
}

required_methylation <- c("patient_id", "study_id", paste0("T_", genes))
missing_methylation <- setdiff(required_methylation, names(methylation))
if (length(missing_methylation) > 0) {
  stop("methylation_wide.tsv missing required columns: ", paste(missing_methylation, collapse = ", "))
}

dat <- merge(clinical, methylation, by = c("patient_id", "study_id"), all = FALSE, sort = FALSE)
if (nrow(dat) != 87) {
  stop("Expected 87 clinical-methylation matched patients, observed ", nrow(dat))
}
if (sum(dat$event == 1, na.rm = TRUE) != 17) {
  stop("Expected 17 recurrence/progression events in all-stage sensitivity data")
}
if (sum(dat$recurrence_primary == 1, na.rm = TRUE) != 82) {
  stop("Expected 82 patients in provider stage I-III/nonpalliative primary subset")
}
if (sum(dat$event == 1 & dat$recurrence_primary == 1, na.rm = TRUE) != 14) {
  stop("Expected 14 events in provider stage I-III/nonpalliative primary subset")
}

write_table <- function(x, path) {
  out <- x
  numeric_cols <- vapply(out, is.double, logical(1))
  out[numeric_cols] <- lapply(out[numeric_cols], function(col) {
    ifelse(is.na(col), "", formatC(col, digits = 17, format = "fg", flag = "#"))
  })
  write.csv(out, path, row.names = FALSE, na = "")
}

run_fit <- function(source, gene, cohort_name) {
  marker_col <- paste0("T_", gene)
  fit_data <- source[, c("patient_id", "study_id", "duration_days", "event", marker_col)]
  names(fit_data)[names(fit_data) == marker_col] <- "marker_pct"
  fit_data$marker_per10 <- fit_data$marker_pct / 10
  fit_data <- fit_data[complete.cases(fit_data[, c("duration_days", "event", "marker_per10")]), ]

  base <- data.frame(
    analysis_population = cohort_name,
    gene = gene,
    n = nrow(source),
    complete_cases = nrow(fit_data),
    events = sum(fit_data$event == 1),
    coef_per10 = NA_real_,
    se_per10 = NA_real_,
    hr_per10 = NA_real_,
    ci_lower = NA_real_,
    ci_upper = NA_real_,
    p_value = NA_real_,
    p_bh = NA_real_,
    ph_chisq = NA_real_,
    ph_p_value = NA_real_,
    ph_violation_p_lt_0_05 = NA,
    estimable = FALSE,
    warnings = "",
    stringsAsFactors = FALSE
  )

  ph_base <- data.frame(
    analysis_population = cohort_name,
    gene = gene,
    variable = "marker_per10",
    rho = NA_real_,
    chisq = NA_real_,
    p_value = NA_real_,
    stringsAsFactors = FALSE
  )

  residual_base <- data.frame(
    analysis_population = character(),
    gene = character(),
    transformed_time = numeric(),
    scaled_schoenfeld_marker_per10 = numeric(),
    stringsAsFactors = FALSE
  )

  if (nrow(fit_data) < 3 || length(unique(fit_data$event)) < 2 || sd(fit_data$marker_per10) == 0) {
    base$warnings <- "Insufficient events/censoring or zero predictor variance"
    return(list(result = base, ph = ph_base, residuals = residual_base))
  }

  warnings_seen <- character()
  fit <- tryCatch(
    withCallingHandlers(
      coxph(Surv(duration_days, event) ~ marker_per10, data = fit_data, ties = "efron", model = TRUE, x = TRUE),
      warning = function(w) {
        warnings_seen <<- c(warnings_seen, conditionMessage(w))
        invokeRestart("muffleWarning")
      }
    ),
    error = function(e) e
  )

  if (inherits(fit, "error")) {
    base$warnings <- paste(c(warnings_seen, conditionMessage(fit)), collapse = " | ")
    return(list(result = base, ph = ph_base, residuals = residual_base))
  }

  s <- summary(fit)
  conf <- suppressWarnings(confint(fit))
  coef_value <- unname(coef(fit)[["marker_per10"]])
  se_value <- unname(s$coefficients["marker_per10", "se(coef)"])
  ci_log <- as.numeric(conf["marker_per10", ])
  hr <- exp(coef_value)
  ci <- exp(ci_log)
  p <- unname(s$coefficients["marker_per10", "Pr(>|z|)"])

  estimable <- all(is.finite(c(coef_value, se_value, hr, ci, p))) &&
    !any(grepl("infinite|not converge|did not converge|ran out of iterations", warnings_seen, ignore.case = TRUE))

  ph <- tryCatch(cox.zph(fit), error = function(e) e)
  residuals_out <- residual_base
  if (!inherits(ph, "error")) {
    ph_table <- as.data.frame(ph$table)
    if ("marker_per10" %in% rownames(ph_table)) {
      if ("rho" %in% colnames(ph$table)) {
        ph_base$rho <- unname(ph$table["marker_per10", "rho"])
      }
      ph_base$chisq <- unname(ph$table["marker_per10", "chisq"])
      ph_base$p_value <- unname(ph$table["marker_per10", "p"])
    }
    y <- ph$y
    if (!is.null(y) && "marker_per10" %in% colnames(y)) {
      residuals_out <- data.frame(
        analysis_population = cohort_name,
        gene = gene,
        transformed_time = as.numeric(rownames(y)),
        scaled_schoenfeld_marker_per10 = as.numeric(y[, "marker_per10"]),
        stringsAsFactors = FALSE
      )
    }
  } else {
    warnings_seen <- c(warnings_seen, paste("cox.zph:", conditionMessage(ph)))
  }

  base$coef_per10 <- if (estimable) coef_value else NA_real_
  base$se_per10 <- if (estimable) se_value else NA_real_
  base$hr_per10 <- if (estimable) hr else NA_real_
  base$ci_lower <- if (estimable) ci[[1]] else NA_real_
  base$ci_upper <- if (estimable) ci[[2]] else NA_real_
  base$p_value <- if (estimable) p else NA_real_
  base$ph_chisq <- ph_base$chisq
  base$ph_p_value <- ph_base$p_value
  base$ph_violation_p_lt_0_05 <- if (is.finite(ph_base$p_value)) ph_base$p_value < 0.05 else NA
  base$estimable <- estimable
  base$warnings <- paste(unique(warnings_seen), collapse = " | ")

  list(result = base, ph = ph_base, residuals = residuals_out)
}

analyze_population <- function(source, cohort_name) {
  fits <- lapply(genes, function(gene) run_fit(source, gene, cohort_name))
  out <- do.call(rbind, lapply(fits, `[[`, "result"))
  out$p_bh <- p.adjust(out$p_value, method = "BH", n = length(genes))
  ph <- do.call(rbind, lapply(fits, `[[`, "ph"))
  residuals <- do.call(rbind, lapply(fits, `[[`, "residuals"))
  list(results = out, ph = ph, residuals = residuals)
}

event_distribution <- function(source, cohort_name) {
  milestones <- sort(unique(c(0, seq(365, max(source$duration_days, na.rm = TRUE), by = 365), max(source$duration_days, na.rm = TRUE))))
  data.frame(
    analysis_population = cohort_name,
    milestone_days = milestones,
    at_risk_at_start = vapply(milestones, function(t) sum(source$duration_days >= t, na.rm = TRUE), integer(1)),
    cumulative_events = vapply(milestones, function(t) sum(source$duration_days <= t & source$event == 1, na.rm = TRUE), integer(1)),
    cumulative_censored = vapply(milestones, function(t) sum(source$duration_days <= t & source$event == 0, na.rm = TRUE), integer(1)),
    stringsAsFactors = FALSE
  )
}

reverse_km_summary <- function(source, cohort_name) {
  fit <- survfit(Surv(duration_days, 1 - event) ~ 1, data = source)
  q <- tryCatch(quantile(fit, probs = c(0.25, 0.5, 0.75)), error = function(e) NULL)
  quant <- if (is.null(q)) c(`25` = NA_real_, `50` = NA_real_, `75` = NA_real_) else q$quantile
  data.frame(
    analysis_population = cohort_name,
    n = nrow(source),
    events = sum(source$event == 1, na.rm = TRUE),
    censored = sum(source$event == 0, na.rm = TRUE),
    min_duration_days = min(source$duration_days, na.rm = TRUE),
    median_observed_duration_days = median(source$duration_days, na.rm = TRUE),
    max_duration_days = max(source$duration_days, na.rm = TRUE),
    reverse_km_q25_followup_days = unname(quant[[1]]),
    reverse_km_median_followup_days = unname(quant[[2]]),
    reverse_km_q75_followup_days = unname(quant[[3]]),
    stringsAsFactors = FALSE
  )
}

primary_data <- dat[dat$recurrence_primary == 1, ]
all_stage_data <- dat

primary <- analyze_population(primary_data, "stage_I_III_nonpalliative_primary")
all_stage <- analyze_population(all_stage_data, "all_87_recurrence_or_progression_sensitivity")

write_table(primary$results, file.path(results_dir, "recurrence.csv"))
write_table(all_stage$results, file.path(results_dir, "recurrence_all_stage.csv"))
write_table(rbind(primary$ph, all_stage$ph), file.path(results_dir, "survival_ph_diagnostics.csv"))
write_table(rbind(primary$residuals, all_stage$residuals), file.path(results_dir, "survival_schoenfeld_residuals.csv"))
write_table(rbind(
  event_distribution(primary_data, "stage_I_III_nonpalliative_primary"),
  event_distribution(all_stage_data, "all_87_recurrence_or_progression_sensitivity")
), file.path(results_dir, "survival_event_time_distribution.csv"))
write_table(rbind(
  reverse_km_summary(primary_data, "stage_I_III_nonpalliative_primary"),
  reverse_km_summary(all_stage_data, "all_87_recurrence_or_progression_sensitivity")
), file.path(results_dir, "survival_followup_summary.csv"))

summary_obj <- list(
  analysis_version_date = "2026-09-05",
  analysis = "Unadjusted Cox proportional hazards models per 10 percentage-point tumor methylation, Efron ties",
  endpoint = "Provider-confirmed recurrence/progression date for events; last follow-up date for non-events",
  primary_population = list(
    name = "stage_I_III_nonpalliative_primary",
    n = nrow(primary_data),
    events = sum(primary_data$event == 1, na.rm = TRUE),
    output = "results/recurrence.csv"
  ),
  sensitivity_population = list(
    name = "all_87_recurrence_or_progression_sensitivity",
    n = nrow(all_stage_data),
    events = sum(all_stage_data$event == 1, na.rm = TRUE),
    output = "results/recurrence_all_stage.csv"
  ),
  diagnostics = list(
    proportional_hazards = "results/survival_ph_diagnostics.csv",
    schoenfeld_residuals = "results/survival_schoenfeld_residuals.csv",
    followup_summary = "results/survival_followup_summary.csv",
    event_time_distribution = "results/survival_event_time_distribution.csv"
  ),
  estimable_primary_models = sum(primary$results$estimable),
  estimable_sensitivity_models = sum(all_stage$results$estimable),
  primary_ph_violations_p_lt_0_05 = primary$results$gene[which(primary$results$ph_violation_p_lt_0_05 %in% TRUE)],
  sensitivity_ph_violations_p_lt_0_05 = all_stage$results$gene[which(all_stage$results$ph_violation_p_lt_0_05 %in% TRUE)],
  primary_min_bh_gene = primary$results$gene[which.min(primary$results$p_bh)],
  primary_min_bh = min(primary$results$p_bh, na.rm = TRUE),
  sensitivity_min_bh_gene = all_stage$results$gene[which.min(all_stage$results$p_bh)],
  sensitivity_min_bh = min(all_stage$results$p_bh, na.rm = TRUE),
  notes = c(
    "Stage is retained as provided by the clinical source; no automatic restaging.",
    "Models are unadjusted exploratory associations only; no risk prediction, C-index, cutoff search, stage adjustment, chemotherapy adjustment, DFS, RFS, or OS endpoint is estimated.",
    "Non-estimable models are left as missing effect estimates and documented by warning text."
  )
)
write_json(summary_obj, file.path(results_dir, "survival_summary.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA)
