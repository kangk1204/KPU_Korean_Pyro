`%||%` <- if (exists("%||%", mode = "function")) get("%||%") else function(a, b) if (is.null(a)) b else a

resampling_get <- function(name, default = NULL) {
  if (exists(name, inherits = TRUE)) get(name, inherits = TRUE) else default
}

resampling_module_path <- function() {
  p <- resampling_get("RECURRENCE_RESAMPLING_MODULE_PATH", NULL)
  if (!is.null(p)) return(p)
  root <- resampling_get("ROOT", getwd())
  file.path(root, "ml", "recurrence_resampling.R")
}

resampling_sha256_file <- function(path) {
  if (exists("sha256_file", mode = "function")) return(sha256_file(path))
  as.character(strsplit(system2("shasum", c("-a", "256", path), stdout = TRUE), " +")[[1]][1])
}

resampling_sha256_text <- function(x) {
  if (exists("sha256_text", mode = "function")) return(sha256_text(x))
  tf <- tempfile()
  writeLines(as.character(x), tf, useBytes = TRUE)
  on.exit(unlink(tf), add = TRUE)
  resampling_sha256_file(tf)
}

resampling_lineage <- function(stage, params = list()) {
  root <- resampling_get("ROOT", getwd())
  input_path <- file.path(root, "data", "ml", "recurrence.tsv")
  config_path <- file.path(root, "registry", "ml_config.json")
  runner_path <- file.path(root, "scripts", "run_recurrence_ml.R")
  metrics_path <- file.path(root, "ml", "recurrence_metrics.R")
  module_path <- resampling_module_path()
  list(
    stage = stage,
    params = params,
    input_hash = if (file.exists(input_path)) resampling_sha256_file(input_path) else NA_character_,
    config_hash = if (file.exists(config_path)) resampling_sha256_file(config_path) else NA_character_,
    runner_hash = if (file.exists(runner_path)) resampling_sha256_file(runner_path) else NA_character_,
    metrics_hash = if (file.exists(metrics_path)) resampling_sha256_file(metrics_path) else NA_character_,
    module_hash = if (file.exists(module_path)) resampling_sha256_file(module_path) else NA_character_
  )
}

resampling_key <- function(stage, iteration, params = list(), lineage = resampling_lineage(stage, params)) {
  payload <- c(lineage, list(iteration = iteration))
  if (requireNamespace("jsonlite", quietly = TRUE)) {
    text <- jsonlite::toJSON(payload, auto_unbox = TRUE, null = "null", digits = NA)
  } else {
    text <- paste(capture.output(str(payload)), collapse = "\n")
  }
  resampling_sha256_text(text)
}

resampling_checkpoint_dir <- function(stage, params = list()) {
  od <- resampling_get("out_dir", file.path(getwd(), "results", "ml", "recurrence"))
  p <- file.path(od, "checkpoints", stage, resampling_key(stage, 0L, params))
  dir.create(p, recursive = TRUE, showWarnings = FALSE)
  p
}

resampling_atomic_save_rds <- function(object, path) {
  tmp <- paste0(path, ".tmp.", Sys.getpid())
  saveRDS(object, tmp)
  if (!file.rename(tmp, path)) {
    unlink(tmp)
    stop("failed to atomically move checkpoint into place: ", path)
  }
  invisible(path)
}

resampling_load_checkpoint <- function(path, expected_key) {
  if (!file.exists(path)) return(NULL)
  x <- readRDS(path)
  if (!identical(x$key, expected_key)) {
    stop("checkpoint key mismatch; refusing to resume nonmatching checkpoint: ", path)
  }
  x
}

resampling_workers <- function() {
  raw <- Sys.getenv("ML_RECURRENCE_WORKERS", "2")
  val <- suppressWarnings(as.integer(raw))
  if (!is.finite(val) || val < 1L) val <- 2L
  min(3L, val)
}

resampling_lapply <- function(X, FUN, ...) {
  workers <- resampling_workers()
  Sys.setenv(
    OMP_NUM_THREADS = "1",
    OPENBLAS_NUM_THREADS = "1",
    MKL_NUM_THREADS = "1",
    VECLIB_MAXIMUM_THREADS = "1",
    NUMEXPR_NUM_THREADS = "1",
    ML_RECURRENCE_NESTED_WORKERS = "1"
  )
  result <- if (.Platform$OS.type != "windows" && workers > 1L && length(X) > 1L) {
    parallel::mclapply(X, FUN, ..., mc.cores = workers, mc.preschedule = FALSE)
  } else {
    lapply(X, FUN, ...)
  }
  failed <- vapply(result, inherits, logical(1), what = "try-error")
  if (any(failed)) stop("Resampling worker failed: ", paste(result[failed], collapse = "\n"))
  result
}

resampling_bind_rows <- function(xs) {
  xs <- xs[!vapply(xs, is.null, logical(1))]
  if (!length(xs)) return(data.frame())
  cols <- unique(unlist(lapply(xs, names)))
  do.call(rbind, lapply(xs, function(x) {
    miss <- setdiff(cols, names(x))
    for (m in miss) x[[m]] <- NA
    x[, cols, drop = FALSE]
  }))
}

resampling_metric_value <- function(aggregate, block, horizon = 1825, metric = "uno_c") {
  z <- aggregate[aggregate$block == block & aggregate$model == "ridge" &
                   aggregate$horizon_days == horizon, , drop = FALSE]
  col <- paste0(metric, "_mean")
  if (nrow(z) != 1L || !(col %in% names(z))) return(NA_real_)
  as.numeric(z[[col]][1])
}

resampling_run_nested_ridge <- function(d, analysis_name, repeats, blocks_override) {
  args <- list(d = d, analysis_name = analysis_name, repeats = repeats,
               models = c("ridge"), blocks_override = blocks_override)
  nested_formals <- names(formals(run_nested))
  if ("n_workers" %in% nested_formals) args$n_workers <- 1L
  if ("save_models" %in% nested_formals) args$save_models <- FALSE
  if ("checkpoint" %in% nested_formals) args$checkpoint <- FALSE
  if ("metric_only" %in% nested_formals) args$metric_only <- TRUE
  do.call(run_nested, args)
}

resampling_permutation_orders <- function(n, B, seed0) {
  lapply(seq_len(B), function(b) {
    set.seed(seed0 + b)
    sample(seq_len(n))
  })
}

run_permutation <- function(d, B = 999) {
  params <- list(B = B, seed = seed, repeats = cfg$uncertainty$permutation_repeats,
                 horizon_days = 1825, block = "combined", model = "ridge")
  checkpoint_dir <- resampling_checkpoint_dir("permutation", params)
  combined_block <- list(combined = model_blocks()$combined)
  orders <- resampling_permutation_orders(nrow(d), B, seed + 909L)
  hashes <- vapply(orders, function(ord) resampling_sha256_text(paste(ord, collapse = ",")), character(1))
  if (length(unique(hashes)) != B) stop("permutation RNG generated duplicate joint permutations")

  observed_key <- resampling_key("permutation_observed", 0L, params)
  observed_path <- file.path(checkpoint_dir, "observed.rds")
  observed_ck <- resampling_load_checkpoint(observed_path, observed_key)
  if (is.null(observed_ck)) {
    obs <- resampling_run_nested_ridge(d, "primary_perm_observed",
                                       cfg$uncertainty$permutation_repeats,
                                       combined_block)
    obs_val <- resampling_metric_value(obs$aggregate, "combined", 1825, "uno_c")
    if (!is.finite(obs_val)) stop("observed permutation statistic is missing or nonfinite")
    observed_ck <- list(key = observed_key, value = obs_val, result = obs$aggregate,
                        lineage = resampling_lineage("permutation_observed", params),
                        status = "complete")
    resampling_atomic_save_rds(observed_ck, observed_path)
  }
  obs_val <- observed_ck$value

  one_perm <- function(b) {
    iter_params <- c(params, list(permutation_hash = hashes[b]))
    lineage <- resampling_lineage("permutation", iter_params)
    key <- resampling_key("permutation", b, iter_params, lineage)
    path <- file.path(checkpoint_dir, sprintf("permutation_%03d.rds", b))
    ck <- resampling_load_checkpoint(path, key)
    if (!is.null(ck)) return(ck$row)

    dp <- d
    ord <- orders[[b]]
    dp$duration_days <- d$duration_days[ord]
    dp$event <- d$event[ord]
    rr <- resampling_run_nested_ridge(dp, sprintf("primary_perm_%03d", b),
                                      cfg$uncertainty$permutation_repeats,
                                      combined_block)
    null_value <- resampling_metric_value(rr$aggregate, "combined", 1825, "uno_c")
    status <- if (is.finite(null_value)) "complete" else "nonestimable"
    row <- data.frame(
      test = "recurrence_combined_ridge_uno_c_5y",
      observed = obs_val,
      permutation = b,
      kind = "null",
      null_value = null_value,
      seed = seed + 909L + b,
      permutation_hash = hashes[b],
      input_hash = lineage$input_hash,
      config_hash = lineage$config_hash,
      runner_hash = lineage$runner_hash,
      metrics_hash = lineage$metrics_hash,
      module_hash = lineage$module_hash,
      checkpoint_key = key,
      status = status,
      stringsAsFactors = FALSE
    )
    resampling_atomic_save_rds(list(key = key, row = row, aggregate = rr$aggregate,
                                    lineage = lineage),
                               path)
    row
  }

  out <- resampling_bind_rows(resampling_lapply(seq_len(B), one_perm))
  if (nrow(out) != B) stop("permutation output count mismatch")
  if (any(out$status != "complete") || any(!is.finite(out$null_value))) {
    stop("permutation produced missing or nonfinite null statistics; refusing p-value without dropping draws")
  }
  out$p_value <- (1 + sum(out$null_value >= obs_val)) / (B + 1)
  out
}

resampling_full_model_row <- function(d, block, features) {
  obj <- fit_ridge_full(d, features, block, inner_groups = NULL, seed0 = seed + 1707L)
  pred <- predict_ridge_full(obj, d)
  oof <- single_eval_oof(d, pred, block, "optimism_full_original", km_source = d)
  agg <- aggregate_metrics(repeat_metrics(oof))
  path <- file.path(model_dir, sprintf("primary_full_original_%s_ridge.rds", block))
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  saveRDS(list(analysis = "primary_full_original", block = block, model = "ridge",
               features = features, fit = obj,
               lineage = resampling_lineage("optimism_full_original",
                                            list(block = block, seed = seed + 1707L))),
          path)
  list(aggregate = agg, model_path = path)
}

resampling_optimism_metrics <- function(apparent, original) {
  merged <- merge(apparent, original,
                  by = c("block", "model", "horizon_days"),
                  suffixes = c("_apparent", "_original"),
                  all = TRUE, sort = FALSE)
  mean_cols <- grep("_mean_apparent$", names(merged), value = TRUE)
  for (col in mean_cols) {
    metric <- sub("_mean_apparent$", "", col)
    orig_col <- paste0(metric, "_mean_original")
    if (orig_col %in% names(merged)) {
      merged[[paste0("optimism_", metric)]] <- merged[[col]] - merged[[orig_col]]
    }
  }
  merged
}

run_optimism <- function(d, B = 500) {
  params <- list(B = B, seed = seed, blocks = names(model_blocks()), model = "ridge")
  checkpoint_dir <- resampling_checkpoint_dir("optimism", params)
  blocks <- model_blocks()

  full_rows <- lapply(names(blocks), function(block) {
    full <- resampling_full_model_row(d, block, blocks[[block]])
    z <- full$aggregate
    z$model_path <- full$model_path
    z
  })
  full_apparent <- resampling_bind_rows(full_rows)

  one_boot <- function(b) {
    set.seed(seed + 707L + b)
    idx <- sample(seq_len(nrow(d)), nrow(d), replace = TRUE)
    original_id <- d$study_id[idx]
    iter_params <- c(params, list(original_ids_hash = resampling_sha256_text(paste(original_id, collapse = ","))))
    lineage <- resampling_lineage("optimism", iter_params)
    key <- resampling_key("optimism", b, iter_params, lineage)
    path <- file.path(checkpoint_dir, sprintf("optimism_%03d.rds", b))
    ck <- resampling_load_checkpoint(path, key)
    if (!is.null(ck)) return(ck$row)

    boot <- d[idx, , drop = FALSE]
    boot$study_id <- paste0(original_id, "_boot", seq_along(idx))
    per_block <- lapply(names(blocks), function(block) {
      obj <- fit_ridge_full(boot, blocks[[block]], block,
                            inner_groups = original_id,
                            seed0 = seed + 707L + b)
      apparent_oof <- single_eval_oof(boot, predict_ridge_full(obj, boot),
                                      block, "optimism_apparent", km_source = boot)
      original_oof <- single_eval_oof(d, predict_ridge_full(obj, d),
                                      block, "optimism_original", km_source = d)
      am <- aggregate_metrics(repeat_metrics(apparent_oof))
      om <- aggregate_metrics(repeat_metrics(original_oof))
      resampling_optimism_metrics(am, om)
    })
    row <- resampling_bind_rows(per_block)
    row$bootstrap <- b
    row$seed <- seed + 707L + b
    row$input_hash <- lineage$input_hash
    row$config_hash <- lineage$config_hash
    row$runner_hash <- lineage$runner_hash
    row$metrics_hash <- lineage$metrics_hash
    row$module_hash <- lineage$module_hash
    row$checkpoint_key <- key
    metric_cols <- grep("^optimism_", names(row), value = TRUE)
    row$status <- if (length(metric_cols) && all(vapply(row[metric_cols], function(x) all(is.finite(x)), logical(1)))) {
      "complete"
    } else {
      "nonestimable"
    }
    resampling_atomic_save_rds(list(key = key, row = row,
                                    lineage = lineage),
                               path)
    row
  }

  boot_rows <- resampling_bind_rows(resampling_lapply(seq_len(B), one_boot))
  if (nrow(unique(boot_rows["bootstrap"])) != B) stop("optimism bootstrap output count mismatch")

  mean_cols <- grep("_mean$", names(full_apparent), value = TRUE)
  corrected <- list()
  ix <- 1L
  for (i in seq_len(nrow(full_apparent))) {
    fa <- full_apparent[i, , drop = FALSE]
    z <- boot_rows[boot_rows$block == fa$block & boot_rows$model == fa$model &
                     boot_rows$horizon_days == fa$horizon_days, , drop = FALSE]
    for (col in mean_cols) {
      metric <- sub("_mean$", "", col)
      opt_col <- paste0("optimism_", metric)
      if (!(opt_col %in% names(z))) next
      vals <- as.numeric(z[[opt_col]])
      ok <- is.finite(vals)
      complete <- length(vals) == B && all(ok) && is.finite(as.numeric(fa[[col]]))
      corrected[[ix]] <- data.frame(
        block = fa$block,
        model = fa$model,
        horizon_days = fa$horizon_days,
        metric = metric,
        original_apparent = as.numeric(fa[[col]]),
        mean_optimism = if (complete) mean(vals) else NA_real_,
        corrected = if (complete) as.numeric(fa[[col]]) - mean(vals) else NA_real_,
        B = B,
        n_estimable = sum(ok),
        status = if (complete) "complete" else "nonestimable",
        model_path = fa$model_path,
        stringsAsFactors = FALSE
      )
      ix <- ix + 1L
    }
  }
  corrected <- resampling_bind_rows(corrected)
  write.csv(corrected, file.path(out_dir, "primary_ridge_optimism_corrected.csv"), row.names = FALSE)
  boot_rows
}
