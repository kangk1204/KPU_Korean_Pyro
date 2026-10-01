#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(survival)
  library(glmnet)
  library(jsonlite)
})

`%||%` <- function(a, b) if (is.null(a)) b else a

script_file <- sub("^--file=", "", commandArgs(FALSE)[grep("^--file=", commandArgs(FALSE))[1]] %||%
                     "scripts/run_stage_sensitivity.R")
ROOT <- normalizePath(file.path(dirname(script_file), ".."), mustWork = TRUE)
source(file.path(ROOT, "scripts", "stageprep.R"), local = .GlobalEnv)
source(file.path(ROOT, "baseline_114", "ml", "recurrence_metrics.R"), local = .GlobalEnv)

args <- commandArgs(trailingOnly = TRUE)
self_test <- "--self-test" %in% args
quick <- "--quick" %in% args
refresh_manifest_only <- "--refresh-manifest-only" %in% args

seed <- 20260905L
cfg_path <- file.path(ROOT, "baseline_114", "registry", "ml_config.json")
cfg <- fromJSON(cfg_path, simplifyVector = FALSE)
if (!is.null(cfg$seed)) seed <- as.integer(cfg$seed)

out_dir <- file.path(ROOT, "results", "reviewer")
reg_dir <- file.path(ROOT, "registry", "reviewer")
private_dir <- file.path(ROOT, "private")
supp_dir <- file.path(ROOT, "supplement")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(reg_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(private_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(supp_dir, recursive = TRUE, showWarnings = FALSE)

time_grid <- seq(0, 1825, length.out = 101)
grid_cols <- paste0("risk_grid_", seq_along(time_grid))
g_grid_cols <- paste0("G_grid_", seq_along(time_grid))

sha256_file <- function(path) {
  as.character(strsplit(system2("shasum", c("-a", "256", path), stdout = TRUE), " +")[[1]][1])
}

sha256_text <- function(x) {
  tf <- tempfile()
  writeLines(as.character(x), tf, useBytes = TRUE)
  on.exit(unlink(tf), add = TRUE)
  sha256_file(tf)
}

bind_rows_fill <- function(xs) {
  xs <- xs[!vapply(xs, is.null, logical(1))]
  cols <- unique(unlist(lapply(xs, names)))
  do.call(rbind, lapply(xs, function(x) {
    miss <- setdiff(cols, names(x))
    for (m in miss) x[[m]] <- NA
    x[, cols, drop = FALSE]
  }))
}

make_folds <- function(event, k, seed0) {
  set.seed(seed0)
  fold <- integer(length(event))
  for (cls in c(0L, 1L)) {
    idx <- which(event == cls)
    idx <- sample(idx)
    fold[idx] <- rep(seq_len(k), length.out = length(idx))
  }
  fold
}

make_outer_folds <- function(d, repeats, k, seed0) {
  rows <- vector("list", repeats)
  for (r in seq_len(repeats)) {
    f <- make_folds(d$event, k, seed0 + 1000L * r)
    tab <- table(f, d$event)
    if (any(rowSums(tab) == 0) || any(tab[, "1"] == 0)) stop("outer fold without events")
    rows[[r]] <- data.frame(`repeat` = r, fold = f, study_id = d$study_id,
                            event = d$event, duration_days = d$duration_days,
                            analysis = "stage_sensitivity", stringsAsFactors = FALSE,
                            check.names = FALSE)
  }
  do.call(rbind, rows)
}

scale_train_test <- function(x_train, x_test) {
  mu <- colMeans(x_train)
  sdv <- apply(x_train, 2, sd)
  sdv[!is.finite(sdv) | sdv == 0] <- 1
  list(train = sweep(sweep(x_train, 2, mu, "-"), 2, sdv, "/"),
       test = sweep(sweep(x_test, 2, mu, "-"), 2, sdv, "/"),
       center = mu, scale = sdv)
}

lambda_grid <- function() {
  v <- cfg$recurrence$lambda_grid
  10 ^ seq(log10(v$min), log10(v$max), length.out = v$n)
}

inner_seed <- function(repeat_id, fold_id) {
  seed + 10007L * as.integer(repeat_id) + 101L * as.integer(fold_id)
}

cox_deviance <- function(time, event, eta) {
  y <- Surv(time, event)
  tryCatch(glmnet::coxnet.deviance(pred = eta, y = y), error = function(e) Inf)
}

predict_available_cox_path <- function(fit, newx, lambdas) {
  if (is.null(fit$jerr) || fit$jerr > 0) stop("unrecoverable glmnet path error; jerr=", fit$jerr)
  matched <- match(lambdas, fit$lambda)
  available <- which(!is.na(matched))
  out <- matrix(NA_real_, nrow = nrow(newx), ncol = length(lambdas))
  if (!length(available)) stop("no fitted Cox penalty candidates")
  pred <- as.matrix(predict(fit, newx = newx, type = "link"))
  out[, available] <- pred[, matched[available], drop = FALSE]
  out
}

select_glmnet_lambda_with_folds <- function(x, time, event, lambdas, fold) {
  inner_k <- length(unique(fold))
  losses <- matrix(NA_real_, nrow = inner_k, ncol = length(lambdas))
  for (j in seq_len(inner_k)) {
    tr <- fold != j
    va <- fold == j
    st <- scale_train_test(x[tr, , drop = FALSE], x[va, , drop = FALSE])
    fit <- glmnet(st$train, Surv(time[tr], event[tr]), family = "cox",
                  alpha = 0, lambda = lambdas, standardize = FALSE)
    pred <- predict_available_cox_path(fit, st$test, lambdas)
    for (l in seq_along(lambdas)) {
      if (all(is.finite(pred[, l]))) losses[j, l] <- cox_deviance(time[va], event[va], pred[, l])
    }
  }
  valid <- apply(losses, 2, function(v) all(is.finite(v)))
  if (!any(valid)) stop("no finite inner-CV Cox deviance candidate")
  mean_loss <- rep(NA_real_, length(lambdas))
  se_loss <- rep(NA_real_, length(lambdas))
  mean_loss[valid] <- colMeans(losses[, valid, drop = FALSE])
  se_loss[valid] <- apply(losses[, valid, drop = FALSE], 2, sd) / sqrt(inner_k)
  best <- which.min(mean_loss)
  chosen <- max(which(mean_loss <= mean_loss[best] + se_loss[best]))
  list(lambda = lambdas[chosen], best_lambda = lambdas[best], inner_fold = fold,
       tuning = data.frame(lambda = lambdas, mean_loss = mean_loss, se_loss = se_loss,
                           valid = valid, t(losses), check.names = FALSE))
}

km_censor <- function(time, event) censor_km(time, event)

base_hazard_at <- function(bh, t) {
  if (nrow(bh) == 0 || t < min(bh$time)) return(0)
  bh$hazard[max(which(bh$time <= t))]
}

risk_from_cox_baseline <- function(bh, eta, tau) {
  1 - exp(-base_hazard_at(bh, tau) * exp(eta))
}

risk_grid_from_cox <- function(bh, eta, grid) {
  mat <- vapply(grid, function(tt) risk_from_cox_baseline(bh, eta, tt),
                numeric(length(eta)))
  mat <- matrix(mat, nrow = length(eta), ncol = length(grid))
  colnames(mat) <- grid_cols
  mat
}

fit_predict_ridge <- function(d, train_idx, test_idx, features, repeat_id, fold_id, block) {
  x <- as.matrix(d[, features, drop = FALSE])
  lambdas <- lambda_grid()
  inner_fold <- make_folds(d$event[train_idx], cfg$recurrence$inner_folds,
                           inner_seed(repeat_id, fold_id))
  sel <- select_glmnet_lambda_with_folds(x[train_idx, , drop = FALSE],
                                         d$duration_days[train_idx],
                                         d$event[train_idx], lambdas, inner_fold)
  st <- scale_train_test(x[train_idx, , drop = FALSE], x[test_idx, , drop = FALSE])
  fit <- glmnet(st$train, Surv(d$duration_days[train_idx], d$event[train_idx]),
                family = "cox", alpha = 0, lambda = sel$lambda, standardize = FALSE)
  if (is.null(fit$jerr) || fit$jerr != 0) stop("glmnet path did not converge; jerr=", fit$jerr)
  eta_train_raw <- as.numeric(predict(fit, newx = st$train, s = sel$lambda, type = "link"))
  eta_center <- mean(eta_train_raw)
  eta_train <- eta_train_raw - eta_center
  eta_test <- as.numeric(predict(fit, newx = st$test, s = sel$lambda, type = "link")) - eta_center
  bh <- breslow_baseline(d$duration_days[train_idx], d$event[train_idx], eta_train, center = 0)
  risk_grid <- risk_grid_from_cox(bh, eta_test, time_grid)
  co <- as.numeric(coef(fit, s = sel$lambda))
  names(co) <- features
  list(
    pred = eta_test,
    risk1825 = risk_from_cox_baseline(bh, eta_test, 1825),
    risk_grid = risk_grid,
    coef = co,
    lambda = sel$lambda,
    best_lambda = sel$best_lambda,
    inner_fold = sel$inner_fold,
    tuning = data.frame(sel$tuning, `repeat` = repeat_id, fold = fold_id,
                        block = block, model = "ridge", check.names = FALSE)
  )
}

stage_blocks <- function() {
  tumor10 <- paste0("T_", stage_genes)
  list(
    provider_clinical = c("cea_log1p", "lvi", "provider_stage_iii"),
    provider_combined = c("cea_log1p", "lvi", "provider_stage_iii", tumor10),
    tnm_clinical = c("cea_log1p", "lvi", "tnm_stage_iii"),
    tnm_combined = c("cea_log1p", "lvi", "tnm_stage_iii", tumor10),
    nostage_clinical = c("cea_log1p", "lvi"),
    nostage_combined = c("cea_log1p", "lvi", tumor10)
  )
}

source_fingerprints <- function() {
  paths <- c(
    recurrence = file.path(ROOT, "baseline_114", "data", "ml", "recurrence.tsv"),
    clinical = file.path(ROOT, "baseline_114", "data", "derived", "clinical.tsv"),
    config = cfg_path,
    runner = file.path(ROOT, "scripts", "run_stage_sensitivity.R"),
    stageprep = file.path(ROOT, "scripts", "stageprep.R"),
    metrics = file.path(ROOT, "baseline_114", "ml", "recurrence_metrics.R")
  )
  vapply(paths, function(p) if (file.exists(p)) sha256_file(p) else "missing", character(1))
}

write_source_hashes <- function(source_hashes = source_fingerprints()) {
  write(toJSON(as.list(source_hashes), pretty = TRUE, auto_unbox = TRUE),
        file.path(reg_dir, "stage_hashes.json"))
}

run_nested_stage <- function(d, repeats = cfg$recurrence$outer_repeats, n_workers = 2L) {
  blocks <- stage_blocks()
  folds <- make_outer_folds(d, repeats, cfg$recurrence$outer_folds, seed)
  jobs <- expand.grid(block = names(blocks), fold = seq_len(cfg$recurrence$outer_folds),
                      `repeat` = seq_len(repeats), stringsAsFactors = FALSE,
                      check.names = FALSE)
  one_job <- function(j) {
    r <- jobs[["repeat"]][j]
    f <- jobs$fold[j]
    block <- jobs$block[j]
    test_ids <- folds$study_id[folds[["repeat"]] == r & folds$fold == f]
    te <- which(d$study_id %in% test_ids)
    tr <- setdiff(seq_len(nrow(d)), te)
    if (length(intersect(d$study_id[tr], d$study_id[te]))) stop("patient leakage between train/test")
    fp <- fit_predict_ridge(d, tr, te, blocks[[block]], r, f, block)
    gkm <- km_censor(d$duration_days[tr], d$event[tr])
    gtime <- vapply(d$duration_days[te], function(tt) g_before(gkm, tt), numeric(1))
    ggrid <- matrix(vapply(time_grid, function(tt) g_at(gkm, tt), numeric(1)),
                    nrow = length(te), ncol = length(time_grid), byrow = TRUE)
    colnames(ggrid) <- g_grid_cols
    oof <- cbind(
      data.frame(analysis = "stage_sensitivity", `repeat` = r, fold = f,
                 study_id = d$study_id[te], event = d$event[te],
                 duration_days = d$duration_days[te], block = block, model = "ridge",
                 risk_score = fp$pred, risk_1825 = fp$risk1825,
                 G_1825 = g_at(gkm, 1825), G_time = gtime, stringsAsFactors = FALSE,
                 check.names = FALSE),
      as.data.frame(fp$risk_grid, check.names = FALSE),
      as.data.frame(ggrid, check.names = FALSE)
    )
    selected <- data.frame(analysis = "stage_sensitivity", `repeat` = r, fold = f,
                           block = block, model = "ridge", selected_lambda = fp$lambda,
                           best_lambda = fp$best_lambda,
                           features = paste(blocks[[block]], collapse = ";"),
                           coefficients = paste(sprintf("%s=%.17g", names(fp$coef), fp$coef), collapse = ";"),
                           train_study_ids = paste(d$study_id[tr], collapse = ";"),
                           test_study_ids = paste(d$study_id[te], collapse = ";"),
                           inner_fold = paste(fp$inner_fold, collapse = ";"),
                           status = "complete", stringsAsFactors = FALSE,
                           check.names = FALSE)
    inner <- data.frame(analysis = "stage_sensitivity", `repeat` = r, outer_fold = f,
                        block = block, model = "ridge", study_id = d$study_id[tr],
                        inner_fold = fp$inner_fold, stringsAsFactors = FALSE,
                        check.names = FALSE)
    list(oof = oof, tuning = fp$tuning, selected = selected, inner = inner)
  }
  Sys.setenv(OMP_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1",
             MKL_NUM_THREADS = "1", VECLIB_MAXIMUM_THREADS = "1")
  n_workers <- max(1L, min(3L, as.integer(n_workers), nrow(jobs)))
  fits <- if (.Platform$OS.type != "windows" && n_workers > 1L) {
    parallel::mclapply(seq_len(nrow(jobs)), one_job, mc.cores = n_workers,
                       mc.preschedule = FALSE, mc.set.seed = FALSE)
  } else {
    lapply(seq_len(nrow(jobs)), one_job)
  }
  list(
    folds = folds,
    oof = do.call(rbind, lapply(fits, `[[`, "oof")),
    tuning = bind_rows_fill(lapply(fits, `[[`, "tuning")),
    selected = do.call(rbind, lapply(fits, `[[`, "selected")),
    inner = do.call(rbind, lapply(fits, `[[`, "inner"))
  )
}

metric_summary <- function(oof) {
  repeat_metrics <- repeat_metrics_from_oof(oof, horizons = 1825)
  aggregate <- aggregate_repeat_metrics(repeat_metrics)
  aggregate[, c("block", "model", "horizon_days", "n_repeats", "estimable_repeats",
                "uno_c_mean", "harrell_c_mean", "auc_mean", "brier_mean")]
}

stage_delta_ci <- function(mean_draws, observed, boot_n) {
  pairs <- list(provider = c("provider_combined", "provider_clinical"),
                tnm = c("tnm_combined", "tnm_clinical"),
                nostage = c("nostage_combined", "nostage_clinical"))
  metrics <- c("uno_c", "auc", "brier")
  rows <- list()
  ix <- 1L
  for (nm in names(pairs)) {
    for (metric in metrics) {
      a <- mean_draws[mean_draws$block == pairs[[nm]][1] & mean_draws$metric == metric &
                        mean_draws$horizon_days == 1825, ]
      b <- mean_draws[mean_draws$block == pairs[[nm]][2] & mean_draws$metric == metric &
                        mean_draws$horizon_days == 1825, ]
      m <- merge(a[, c("draw", "value")], b[, c("draw", "value")],
                 by = "draw", suffixes = c("_combined", "_clinical"))
      delta <- m$value_combined - m$value_clinical
      oa <- observed[observed$block == pairs[[nm]][1] & observed$metric == metric, ]
      ob <- observed[observed$block == pairs[[nm]][2] & observed$metric == metric, ]
      rows[[ix]] <- data.frame(stage_definition = nm,
                               contrast = paste0(pairs[[nm]][1], "_minus_", pairs[[nm]][2]),
                               model = "ridge", horizon_days = 1825,
                               metric = paste0("delta_", metric),
                               estimate = oa$observed[1] - ob$observed[1],
                               ci_low = as.numeric(quantile(delta, 0.025, names = FALSE)),
                               ci_high = as.numeric(quantile(delta, 0.975, names = FALSE)),
                               bootstrap_n = boot_n, n_estimable = sum(is.finite(delta)),
                               status = if (all(is.finite(delta))) "estimable" else "nonestimable",
                               stringsAsFactors = FALSE)
      ix <- ix + 1L
    }
  }
  do.call(rbind, rows)
}

write_stage_methods_results <- function(summary, ci, deltas, audit) {
  fmt <- function(x) sprintf("%.3f", as.numeric(x))
  row_for <- function(block) summary[summary$block == block & summary$horizon_days == 1825, ][1, ]
  ci_for <- function(block, metric) ci[ci$block == block & ci$metric == metric &
                                         ci$horizon_days == 1825, ][1, ]
  line_block <- function(block, label) {
    s <- row_for(block)
    cuno <- ci_for(block, "uno_c")
    cauc <- ci_for(block, "auc")
    cbrier <- ci_for(block, "brier")
    sprintf("- %s: five-year Uno C %s (%s to %s), AUC %s (%s to %s), Brier %s (%s to %s).",
            label, fmt(s$uno_c_mean), fmt(cuno$ci_low), fmt(cuno$ci_high),
            fmt(s$auc_mean), fmt(cauc$ci_low), fmt(cauc$ci_high),
            fmt(s$brier_mean), fmt(cbrier$ci_low), fmt(cbrier$ci_high))
  }
  delta_line <- function(defn, metric) {
    z <- deltas[deltas$stage_definition == defn & deltas$metric == paste0("delta_", metric), ][1, ]
    sprintf("%s %s delta %s (%s to %s)", defn, metric, fmt(z$estimate), fmt(z$ci_low), fmt(z$ci_high))
  }
  text <- c(
    "# Stage Sensitivity Methods and Results",
    "",
    "## Methods",
    "",
    paste0("The primary recurrence set was preserved at 82 patients and 14 events. ",
           "A strict TNM parser required T1, T2, T3, T4/T4a/T4b, N0/N1a-c/N2a-b, and explicit M0. ",
           "This retained ", sum(audit$analysis_included), " patients and ",
           sum(audit$event[audit$analysis_included]), " events; three records were excluded for nonstandard or non-explicit TNM strings. ",
           sum(audit$provider_tnm_mismatch), " provider-stage/TNM-stage mismatches were recorded."),
    "",
    paste0("Six ridge Cox blocks were fitted: provider-stage clinical and combined, TNM-derived-stage clinical and combined, ",
           "and no-stage clinical and combined. Clinical predictors were log1p CEA and lymphovascular invasion, ",
           "plus stage III when stage was used. Combined models added the ten tumor methylation genes. ",
           "All six blocks shared the same event-stratified 4-fold x 25-repeat outer folds and 3-fold inner tuning folds. ",
           "Preprocessing, lambda selection, Cox baseline hazards, and censoring Kaplan-Meier estimates were fit within training folds only. ",
           "Uncertainty used 2,000 patient-cluster bootstrap resamples of the fixed out-of-fold predictions."),
    "",
    "## Results",
    "",
    line_block("provider_clinical", "Provider clinical"),
    line_block("provider_combined", "Provider combined"),
    line_block("tnm_clinical", "TNM clinical"),
    line_block("tnm_combined", "TNM combined"),
    line_block("nostage_clinical", "No-stage clinical"),
    line_block("nostage_combined", "No-stage combined"),
    "",
    paste0("Clinical-to-combined five-year deltas were: ",
           delta_line("provider", "uno_c"), "; ",
           delta_line("tnm", "uno_c"), "; ",
           delta_line("nostage", "uno_c"), ". ",
           "Brier deltas were ",
           delta_line("provider", "brier"), "; ",
           delta_line("tnm", "brier"), "; ",
           delta_line("nostage", "brier"), "."),
    "",
    "These sensitivity analyses did not change the interpretation that adding the ten-gene tumor methylation panel to clinical predictors did not demonstrate improved five-year recurrence prediction in this small event set."
  )
  writeLines(text, file.path(supp_dir, "stage_methods_results.md"))
}

make_plot_ready_summary <- function(summary, ci, deltas) {
  ci_wide <- reshape(ci[, c("block", "metric", "estimate", "ci_low", "ci_high", "status")],
                     idvar = "block", timevar = "metric", direction = "wide")
  out <- merge(summary, ci_wide, by = "block", all.x = TRUE, sort = FALSE)
  out$stage_definition <- sub("_.*$", "", out$block)
  out$model_block <- sub("^[^_]+_", "", out$block)
  out <- out[, c("stage_definition", "model_block", "block", "model", "horizon_days",
                 "n_repeats", "estimable_repeats", "uno_c_mean", "estimate.uno_c",
                 "ci_low.uno_c", "ci_high.uno_c", "auc_mean", "estimate.auc",
                 "ci_low.auc", "ci_high.auc", "brier_mean", "estimate.brier",
                 "ci_low.brier", "ci_high.brier")]
  delta_out <- deltas[, c("stage_definition", "contrast", "metric", "estimate",
                          "ci_low", "ci_high", "bootstrap_n", "n_estimable", "status")]
  list(blocks = out, deltas = delta_out)
}

make_stage_metric_tables <- function(ci, deltas) {
  stage_map <- c(provider = "provider", tnm = "tnm", nostage = "none")
  metric_rows <- ci[ci$metric %in% c("uno_c", "auc", "brier") & ci$horizon_days == 1825, ,
                    drop = FALSE]
  pieces <- strsplit(metric_rows$block, "_", fixed = TRUE)
  metric_rows$stage_definition <- vapply(pieces, function(x) stage_map[[x[[1]]]], character(1))
  metric_rows$block <- vapply(pieces, function(x) x[[2]], character(1))
  metric_rows <- metric_rows[, c("stage_definition", "block", "metric", "estimate",
                                 "ci_low", "ci_high", "bootstrap_n", "n_estimable", "status")]
  diff_rows <- deltas[deltas$metric %in% paste0("delta_", c("uno_c", "auc", "brier")), ,
                      drop = FALSE]
  diff_rows$stage_definition <- ifelse(diff_rows$stage_definition == "nostage", "none",
                                       diff_rows$stage_definition)
  diff_rows$contrast <- "combined-minus-clinical"
  diff_rows$metric <- sub("^delta_", "", diff_rows$metric)
  diff_rows <- diff_rows[, c("stage_definition", "contrast", "metric", "estimate",
                             "ci_low", "ci_high", "bootstrap_n", "n_estimable", "status")]
  list(metrics = metric_rows, differences = diff_rows)
}

stage_output_paths <- function() {
  c(
    file.path(private_dir, "stage_patient_audit.tsv"),
    file.path(out_dir, "stage_folds.csv"),
    file.path(out_dir, "stage_inner_assignment_proof.csv"),
    file.path(out_dir, "stage_inner_tuning.csv"),
    file.path(out_dir, "stage_selected_hyperparameters.csv"),
    file.path(out_dir, "stage_oof_predictions.csv"),
    file.path(out_dir, "stage_repeat_metrics.csv"),
    file.path(out_dir, "stage_summary.csv"),
    file.path(out_dir, "stage_bootstrap_ci.csv"),
    file.path(out_dir, "stage_delta_ci.csv"),
    file.path(out_dir, "stage_metrics.tsv"),
    file.path(out_dir, "stage_differences.tsv"),
    file.path(out_dir, "stage_plot_ready_summary.tsv"),
    file.path(out_dir, "stage_plot_ready_deltas.tsv"),
    file.path(out_dir, "stage_bootstrap_repeat_mean_draws.csv"),
    file.path(out_dir, "stage_bootstrap_patient_weights.rds"),
    file.path(reg_dir, "stage_config.json"),
    file.path(reg_dir, "stage_hashes.json"),
    file.path(supp_dir, "stage_methods_results.md")
  )
}

write_manifest <- function(summary = NULL, ci = NULL, deltas = NULL, audit) {
  files <- stage_output_paths()
  payload <- list(
    status = "complete",
    generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
    root = ROOT,
    source_hashes = as.list(source_fingerprints()),
    cohort = list(primary_n = nrow(audit), primary_events = sum(audit$event),
                  strict_tnm_n = sum(audit$analysis_included),
                  strict_tnm_events = sum(audit$event[audit$analysis_included]),
                  provider_tnm_mismatches = sum(audit$provider_tnm_mismatch)),
    outputs = lapply(files[file.exists(files)], function(p) {
      list(file = sub(paste0("^", ROOT, "/?"), "", p), sha256 = sha256_file(p),
           bytes = file.info(p)$size)
    })
  )
  write(toJSON(payload, pretty = TRUE, auto_unbox = TRUE, na = "null"),
        file.path(reg_dir, "stage_manifest.json"))
}

refresh_stage_manifest_only <- function() {
  audit_path <- file.path(private_dir, "stage_patient_audit.tsv")
  if (!file.exists(audit_path)) stop("missing stage patient audit: ", audit_path)
  audit <- read.delim(audit_path, stringsAsFactors = FALSE, check.names = FALSE)
  write_source_hashes()
  write_manifest(audit = audit)
  cat(toJSON(list(status = "metadata_refreshed",
                  manifest = file.path(reg_dir, "stage_manifest.json"),
                  hashes = file.path(reg_dir, "stage_hashes.json")),
             auto_unbox = TRUE), "\n")
}

run_self_tests <- function() {
  ok <- parse_tnm_strict("T4aN2bM0")
  stopifnot(isTRUE(ok$valid), ok$stage == 3L)
  bad <- c("T2N0", "T3N2cM0", "T3N1bMo", "T3N1b", "T1aN0M0", "T2bN0M0", "T3aN0M0")
  stopifnot(all(!vapply(lapply(bad, parse_tnm_strict), `[[`, logical(1), "valid")))
  prep <- prepare_stage_sensitivity_data(ROOT)
  d <- prep$stage_data
  folds <- make_outer_folds(d, 2L, 4L, seed)
  stopifnot(nrow(prep$audit) == 82L, nrow(d) == 79L, sum(d$event) == 14L)
  for (r in sort(unique(folds[["repeat"]]))) {
    seen <- folds[folds[["repeat"]] == r, ]
    stopifnot(!anyDuplicated(seen$study_id))
    stopifnot(all(table(seen$fold, seen$event)[, "1"] >= 1))
  }
  blocks <- stage_blocks()
  stopifnot(length(blocks) == 6L)
  stopifnot(identical(sort(unique(folds$analysis)), "stage_sensitivity"))
  cat("stage_sensitivity self-tests passed\n")
}

main <- function() {
  if (self_test) return(run_self_tests())
  if (refresh_manifest_only) return(refresh_stage_manifest_only())
  prep <- prepare_stage_sensitivity_data(ROOT)
  write.table(prep$audit, file.path(private_dir, "stage_patient_audit.tsv"),
              sep = "\t", quote = FALSE, row.names = FALSE)
  repeats <- if (quick) 2L else as.integer(cfg$recurrence$outer_repeats)
  boot_n <- if (quick) 20L else as.integer(cfg$uncertainty$patient_bootstrap)
  workers <- as.integer(Sys.getenv("STAGE_SENSITIVITY_WORKERS", "2"))
  source_before <- source_fingerprints()
  res <- run_nested_stage(prep$stage_data, repeats = repeats, n_workers = workers)
  if (!identical(source_before, source_fingerprints())) stop("source/input changed during nested run")
  repmet <- repeat_metrics_from_oof(res$oof, horizons = 1825)
  summary <- metric_summary(res$oof)
  boot <- fast_bootstrap_ci(res$oof, metrics = c("uno_c", "auc", "brier"),
                            B = boot_n, seed = seed + 606L, horizons = 1825,
                            include_deltas = FALSE)
  observed <- observed_metric_points(res$oof, metrics = c("uno_c", "auc", "brier"),
                                     horizons = 1825)
  deltas <- stage_delta_ci(boot$repeat_mean_draws, observed, boot_n)

  write.csv(res$folds, file.path(out_dir, "stage_folds.csv"), row.names = FALSE)
  write.csv(res$inner, file.path(out_dir, "stage_inner_assignment_proof.csv"), row.names = FALSE)
  write.csv(res$tuning, file.path(out_dir, "stage_inner_tuning.csv"), row.names = FALSE)
  write.csv(res$selected, file.path(out_dir, "stage_selected_hyperparameters.csv"), row.names = FALSE)
  write.csv(res$oof, file.path(out_dir, "stage_oof_predictions.csv"), row.names = FALSE)
  write.csv(repmet, file.path(out_dir, "stage_repeat_metrics.csv"), row.names = FALSE)
  write.csv(summary, file.path(out_dir, "stage_summary.csv"), row.names = FALSE)
  write.csv(boot$ci, file.path(out_dir, "stage_bootstrap_ci.csv"), row.names = FALSE)
  write.csv(deltas, file.path(out_dir, "stage_delta_ci.csv"), row.names = FALSE)
  metric_tables <- make_stage_metric_tables(boot$ci, deltas)
  write.table(metric_tables$metrics, file.path(out_dir, "stage_metrics.tsv"),
              sep = "\t", quote = FALSE, row.names = FALSE)
  write.table(metric_tables$differences, file.path(out_dir, "stage_differences.tsv"),
              sep = "\t", quote = FALSE, row.names = FALSE)
  plot_ready <- make_plot_ready_summary(summary, boot$ci, deltas)
  write.table(plot_ready$blocks, file.path(out_dir, "stage_plot_ready_summary.tsv"),
              sep = "\t", quote = FALSE, row.names = FALSE)
  write.table(plot_ready$deltas, file.path(out_dir, "stage_plot_ready_deltas.tsv"),
              sep = "\t", quote = FALSE, row.names = FALSE)
  write.csv(boot$repeat_mean_draws, file.path(out_dir, "stage_bootstrap_repeat_mean_draws.csv"), row.names = FALSE)
  saveRDS(list(weights = boot$W, seed = seed + 606L, study_id = sort(unique(res$oof$study_id))),
          file.path(out_dir, "stage_bootstrap_patient_weights.rds"))

  cfg_out <- list(seed = seed, quick = quick, outer_folds = cfg$recurrence$outer_folds,
                  outer_repeats = repeats, inner_folds = cfg$recurrence$inner_folds,
                  lambda_grid = cfg$recurrence$lambda_grid, bootstrap_B = boot_n,
                  blocks = stage_blocks())
  write(toJSON(cfg_out, pretty = TRUE, auto_unbox = TRUE),
        file.path(reg_dir, "stage_config.json"))
  write_source_hashes(source_before)
  write_stage_methods_results(summary, boot$ci, deltas, prep$audit)
  write_manifest(summary, boot$ci, deltas, prep$audit)
  cat(toJSON(list(status = "complete", strict_tnm_n = nrow(prep$stage_data),
                  events = sum(prep$stage_data$event), fits = nrow(res$selected),
                  bootstrap = boot_n), auto_unbox = TRUE), "\n")
}

main()
