library(testthat)

runner_path <- file.path("scripts", "run_recurrence_ml.R")
if (!file.exists(runner_path)) {
  cmd_args <- commandArgs(FALSE)
  file_arg <- cmd_args[grepl("^--file=", cmd_args)]
  script_dir <- if (length(file_arg)) dirname(normalizePath(sub("^--file=", "", file_arg[1]), mustWork = FALSE)) else "."
  runner_path <- file.path(script_dir, "..", "scripts", "run_recurrence_ml.R")
}
RECURRENCE_RUNNER_PATH <- normalizePath(runner_path, mustWork = TRUE)

load_recurrence_runner <- function() {
  old_source_only <- Sys.getenv("ML_RECURRENCE_SOURCE_ONLY", unset = NA_character_)
  old_global_names <- ls(envir = .GlobalEnv, all.names = TRUE)
  old_wd <- getwd()
  env <- new.env(parent = .GlobalEnv)
  Sys.setenv(ML_RECURRENCE_SOURCE_ONLY = "1")
  on.exit({
    setwd(old_wd)
    if (is.na(old_source_only)) Sys.unsetenv("ML_RECURRENCE_SOURCE_ONLY") else
      Sys.setenv(ML_RECURRENCE_SOURCE_ONLY = old_source_only)
    added <- setdiff(ls(envir = .GlobalEnv, all.names = TRUE), old_global_names)
    if (length(added)) rm(list = added, envir = .GlobalEnv)
  }, add = TRUE)
  setwd(normalizePath(file.path(dirname(RECURRENCE_RUNNER_PATH), ".."), mustWork = TRUE))
  source(RECURRENCE_RUNNER_PATH, local = env)
  added <- setdiff(ls(envir = .GlobalEnv, all.names = TRUE), old_global_names)
  for (name in added) {
    value <- get(name, envir = .GlobalEnv)
    if (is.function(value)) environment(value) <- env
    assign(name, value, envir = env)
  }
  env
}

reconstruct_permutation_fold3 <- function(env) {
  d0 <- env$read_input()
  primary <- d0[d0$recurrence_primary == 1, , drop = FALSE]
  set.seed(env$seed + 909L + 1L)
  order <- sample(seq_len(nrow(primary)))
  permuted <- primary
  permuted$duration_days <- primary$duration_days[order]
  permuted$event <- primary$event[order]
  folds <- env$make_outer_folds(
    permuted,
    env$cfg$uncertainty$permutation_repeats,
    env$cfg$recurrence$outer_folds,
    env$seed,
    "primary_perm_001"
  )
  features <- env$model_blocks()$combined
  test_ids <- folds$study_id[folds[["repeat"]] == 1L & folds$fold == 3L]
  test_idx <- which(permuted$study_id %in% test_ids)
  train_idx <- setdiff(seq_len(nrow(permuted)), test_idx)
  list(d = permuted, train_idx = train_idx, test_idx = test_idx, features = features)
}

test_that("permutation 1 fold 3 reproduces the nonfatal glmnet Cox path truncation", {
  env <- load_recurrence_runner()
  fixture <- reconstruct_permutation_fold3(env)
  d <- fixture$d
  train_idx <- fixture$train_idx
  features <- fixture$features
  x <- as.matrix(d[, features, drop = FALSE])
  lambdas <- env$lambda_grid()
  inner <- env$make_folds(
    d$event[train_idx],
    env$cfg$recurrence$inner_folds,
    env$inner_seed(1L, 3L)
  )

  expect_equal(length(train_idx), 62)
  expect_equal(length(fixture$test_idx), 20)
  expect_equal(sum(d$event[train_idx]), 11)
  expect_equal(sum(d$event[fixture$test_idx]), 3)
  expect_equal(lambdas[1], 1e-04, tolerance = 1e-15)
  expect_equal(tail(lambdas, 1), 100, tolerance = 1e-12)

  inner_fold <- 2L
  inner_train <- inner != inner_fold
  inner_valid <- inner == inner_fold
  st <- env$scale_train_test(
    x[train_idx, , drop = FALSE][inner_train, , drop = FALSE],
    x[train_idx, , drop = FALSE][inner_valid, , drop = FALSE]
  )
  fit <- suppressWarnings(glmnet::glmnet(
    st$train,
    survival::Surv(d$duration_days[train_idx][inner_train], d$event[train_idx][inner_train]),
    family = "cox",
    alpha = 0,
    lambda = lambdas,
    standardize = FALSE
  ))

  expect_equal(fit$jerr, -30025)
  expect_false(glmnet:::jerr.coxnet(fit$jerr, 100000, 1000)$fatal)
  expect_equal(which(is.na(match(lambdas, fit$lambda))), 1L)
  expect_equal(sort(match(fit$lambda, lambdas)), 2:25)

  path_pred <- suppressWarnings(env$predict_available_cox_path(fit, st$test, lambdas))
  expect_true(all(is.na(path_pred[, 1])))
  expect_true(all(is.finite(path_pred[, -1])))
})

test_that("selector excludes the unavailable smallest lambda and final selected fit remains strict", {
  env <- load_recurrence_runner()
  fixture <- reconstruct_permutation_fold3(env)
  d <- fixture$d
  train_idx <- fixture$train_idx
  test_idx <- fixture$test_idx
  features <- fixture$features
  x <- as.matrix(d[, features, drop = FALSE])
  lambdas <- env$lambda_grid()

  selected <- suppressWarnings(env$select_glmnet_lambda(
    x[train_idx, , drop = FALSE],
    d$duration_days[train_idx],
    d$event[train_idx],
    0,
    lambdas,
    env$cfg$recurrence$inner_folds,
    env$inner_seed(1L, 3L)
  ))

  expect_false(selected$tuning$valid[1])
  expect_true(all(selected$tuning$valid[-1]))
  expect_equal(selected$lambda, 100, tolerance = 1e-12)
  expect_equal(selected$best_lambda, 100, tolerance = 1e-12)

  fp <- suppressWarnings(env$fit_predict_glmnet(
    d, train_idx, test_idx, features, 0, 1L, 3L, "combined", "ridge"
  ))
  expect_equal(fp$fit$model_object$jerr, 0)
  expect_equal(length(fp$pred), length(test_idx))
  expect_true(all(is.finite(fp$pred)))
  expect_true(all(is.finite(fp$risk1095)))
  expect_true(all(is.finite(fp$risk1825)))
  expect_true(all(fp$risk1095 >= 0 & fp$risk1095 <= 1))
  expect_true(all(fp$risk1825 >= 0 & fp$risk1825 <= 1))
})

test_that("outer artifact schema keeps stable recurrence runner row keys", {
  env <- load_recurrence_runner()
  fixture <- reconstruct_permutation_fold3(env)
  d <- fixture$d
  blocks <- list(combined = fixture$features)
  result <- suppressWarnings(env$run_nested(
    d,
    analysis_name = "primary_perm_001_schema_regression",
    repeats = 1L,
    models = "ridge",
    blocks_override = blocks,
    n_workers = 1L,
    save_models = FALSE,
    checkpoint = FALSE,
    metric_only = TRUE
  ))

  expect_true(all(c("analysis", "repeat", "fold", "study_id", "event", "duration_days",
                    "block", "model", "risk_score", "risk_1095", "risk_1825",
                    "G_1095", "G_1825", "G_time") %in% names(result$oof)))
  expect_true(all(c("analysis", "repeat", "fold", "block", "model", "selected_lambda",
                    "status", "warning_count") %in% names(result$model_summary)))
  expect_true(all(c("lambda", "mean_loss", "se_loss", "valid", "repeat", "fold",
                    "block", "model") %in% names(result$tuning)))
  expect_equal(unique(result$oof$analysis), "primary_perm_001_schema_regression")
  expect_equal(unique(result$oof$block), "combined")
  expect_equal(unique(result$oof$model), "ridge")
  expect_equal(unname(as.integer(sort(table(result$oof$study_id)))), rep(1L, nrow(d)))
})
