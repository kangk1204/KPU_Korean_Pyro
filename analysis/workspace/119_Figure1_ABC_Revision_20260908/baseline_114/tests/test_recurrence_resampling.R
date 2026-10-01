library(testthat)

module_path <- file.path("ml", "recurrence_resampling.R")
if (!file.exists(module_path)) {
  cmd_args <- commandArgs(FALSE)
  file_arg <- cmd_args[grepl("^--file=", cmd_args)]
  script_dir <- if (length(file_arg)) dirname(normalizePath(sub("^--file=", "", file_arg[1]), mustWork = FALSE)) else "."
  module_path <- file.path(script_dir, "..", "ml", "recurrence_resampling.R")
}
RECURRENCE_RESAMPLING_MODULE_PATH <- normalizePath(module_path, mustWork = TRUE)
source(RECURRENCE_RESAMPLING_MODULE_PATH)

make_fixture <- function() {
  data.frame(
    study_id = paste0("P", 1:8),
    duration_days = c(400, 900, 1200, 1600, 500, 1000, 1500, 1900),
    event = c(1, 0, 1, 0, 0, 1, 0, 0),
    stage_iii = c(1, 0, 1, 0, 0, 1, 0, 0),
    cea_log1p = seq(0.1, 0.8, length.out = 8),
    lvi = c(1, 0, 0, 1, 0, 1, 0, 0),
    T_EYA4 = seq(1, 8),
    T_ZNF568 = seq(8, 1),
    stringsAsFactors = FALSE
  )
}

install_stubs <- function(tmp) {
  assign("ROOT", tmp, envir = .GlobalEnv)
  tmp_out <- file.path(tmp, "results", "ml", "recurrence")
  assign("out_dir", tmp_out, envir = .GlobalEnv)
  assign("model_dir", file.path(tmp_out, "models"), envir = .GlobalEnv)
  dir.create(file.path(tmp, "data", "ml"), recursive = TRUE, showWarnings = FALSE)
  dir.create(file.path(tmp, "registry"), recursive = TRUE, showWarnings = FALSE)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)
  writeLines("input", file.path(tmp, "data", "ml", "recurrence.tsv"))
  writeLines("{}", file.path(tmp, "registry", "ml_config.json"))
  dir.create(file.path(tmp, "scripts"), recursive = TRUE, showWarnings = FALSE)
  dir.create(file.path(tmp, "ml"), recursive = TRUE, showWarnings = FALSE)
  file.copy(RECURRENCE_RESAMPLING_MODULE_PATH, file.path(tmp, "ml", "recurrence_resampling.R"), overwrite = TRUE)
  writeLines("runner", file.path(tmp, "scripts", "run_recurrence_ml.R"))
  writeLines("metrics", file.path(tmp, "ml", "recurrence_metrics.R"))
  assign("seed", 20260905, envir = .GlobalEnv)
  assign("cfg", list(uncertainty = list(permutation_repeats = 2)), envir = .GlobalEnv)
  Sys.setenv(ML_RECURRENCE_WORKERS = "1")

  assign("model_blocks", function() {
    list(
      clinical = c("stage_iii", "cea_log1p", "lvi"),
      tumor10 = c("T_EYA4", "T_ZNF568"),
      combined = c("stage_iii", "cea_log1p", "lvi", "T_EYA4", "T_ZNF568")
    )
  }, envir = .GlobalEnv)
  assign("run_nested", function(d, analysis_name, repeats, models, blocks_override = NULL,
                          n_workers = NULL, save_models = TRUE, checkpoint = TRUE,
                          metric_only = FALSE) {
    if (!identical(n_workers, 1L) || !identical(save_models, FALSE) ||
        !identical(checkpoint, FALSE) || !identical(metric_only, TRUE)) {
      stop("run_nested called without resampling-safe arguments")
    }
    val <- (sum(d$event * seq_len(nrow(d))) + repeats + length(models)) / 100
    list(aggregate = data.frame(block = names(blocks_override)[1], model = "ridge",
                                horizon_days = 1825, n_repeats = repeats,
                                uno_c_mean = val))
  }, envir = .GlobalEnv)
  assign("fit_ridge_full", function(d, features, block, inner_groups = NULL, seed0 = seed) {
    list(block = block, features = features, event_sum = sum(d$event),
         duplicated_groups = !is.null(inner_groups) && any(duplicated(inner_groups)),
         seed0 = seed0)
  }, envir = .GlobalEnv)
  assign("predict_ridge_full", function(obj, d) {
    list(value = obj$event_sum / nrow(d), score = rep(obj$event_sum / nrow(d), nrow(d)))
  }, envir = .GlobalEnv)
  assign("single_eval_oof", function(d, pred, block, analysis, km_source = d) {
    bump <- if (analysis == "optimism_apparent") 0.20 else if (analysis == "optimism_original") 0.05 else 0
    data.frame(block = block, model = "ridge", horizon_days = c(1095, 1825),
               uno_c_mean = pred$value + bump,
               brier_mean = 1 - pred$value + bump,
               stringsAsFactors = FALSE)
  }, envir = .GlobalEnv)
  assign("repeat_metrics", function(oof) oof, envir = .GlobalEnv)
  assign("aggregate_metrics", function(repmet) repmet, envir = .GlobalEnv)
}

test_that("permutation uses unique deterministic orders and exact checkpoints", {
  tmp <- tempfile()
  install_stubs(tmp)
  d <- make_fixture()

  first <- run_permutation(d, B = 3)
  second <- run_permutation(d, B = 3)

  expect_equal(nrow(first), 3)
  expect_equal(first$permutation_hash, second$permutation_hash)
  expect_equal(length(unique(first$permutation_hash)), 3)
  expect_true(all(first$status == "complete"))
  expect_true(all(is.finite(first$null_value)))
  expect_true(all(first$p_value == (1 + sum(first$null_value >= first$observed[1])) / 4))
  checkpoints <- list.files(file.path(out_dir, "checkpoints", "permutation"),
                            pattern = "^permutation_[0-9]+[.]rds$", recursive = TRUE)
  expect_equal(sort(basename(checkpoints)), sprintf("permutation_%03d.rds", 1:3))
})

test_that("permutation refuses nonfinite null statistics", {
  tmp <- tempfile()
  install_stubs(tmp)
  d <- make_fixture()
  assign("run_nested", function(d, analysis_name, repeats, models, blocks_override = NULL,
                          n_workers = NULL, save_models = TRUE, checkpoint = TRUE,
                          metric_only = FALSE) {
    val <- if (analysis_name == "primary_perm_001") NA_real_ else 0.4
    list(aggregate = data.frame(block = names(blocks_override)[1], model = "ridge",
                                horizon_days = 1825, n_repeats = repeats,
                                uno_c_mean = val))
  }, envir = .GlobalEnv)

  expect_error(run_permutation(d, B = 2), "missing or nonfinite null")
})

test_that("optimism writes full-original corrected points and models", {
  tmp <- tempfile()
  install_stubs(tmp)
  d <- make_fixture()

  boot <- run_optimism(d, B = 4)
  corrected <- read.csv(file.path(out_dir, "primary_ridge_optimism_corrected.csv"))

  expect_equal(length(unique(boot$bootstrap)), 4)
  expect_true(all(boot$status == "complete"))
  expect_true(all(c("clinical", "tumor10", "combined") %in% corrected$block))
  target <- corrected[corrected$block == "combined" & corrected$horizon_days == 1825 &
                        corrected$metric == "uno_c", ]
  draws <- boot[boot$block == "combined" & boot$horizon_days == 1825, ]
  expect_equal(target$corrected, target$original_apparent - mean(draws$optimism_uno_c),
               tolerance = 1e-12)
  expect_equal(target$B, 4)
  expect_equal(target$status, "complete")
  expect_true(file.exists(file.path(model_dir, "primary_full_original_clinical_ridge.rds")))
  expect_true(file.exists(file.path(model_dir, "primary_full_original_tumor10_ridge.rds")))
  expect_true(file.exists(file.path(model_dir, "primary_full_original_combined_ridge.rds")))
})
