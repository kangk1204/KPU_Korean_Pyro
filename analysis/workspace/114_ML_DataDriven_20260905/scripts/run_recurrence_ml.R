#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(survival)
  library(glmnet)
  library(ranger)
  library(jsonlite)
})

`%||%` <- function(a, b) if (is.null(a)) b else a
script_file <- sub("^--file=", "", commandArgs(FALSE)[grep("^--file=", commandArgs(FALSE))[1]] %||% "scripts/run_recurrence_ml.R")
ROOT <- normalizePath(file.path(dirname(script_file), ".."), mustWork = FALSE)
if (!dir.exists(file.path(ROOT, "data"))) ROOT <- normalizePath(getwd(), mustWork = TRUE)

genes <- c("EYA4", "ZNF568", "ZNF793", "SFMBT2", "ADHFE1", "HOXA2", "BEND5", "UNC5C", "RALYL", "GFRA1")
out_dir <- file.path(ROOT, "results", "ml", "recurrence")
ver_dir <- file.path(ROOT, "verification", "ml")
model_dir <- file.path(ROOT, "results", "ml", "recurrence", "models")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(ver_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)

args <- commandArgs(trailingOnly = TRUE)
quick <- "--quick" %in% args
self_test <- "--self-test" %in% args
stage_arg <- args[grepl("^--stage=", args)]
stage <- if (length(stage_arg)) sub("^--stage=", "", stage_arg[1]) else "all"
if (quick) out_dir <- file.path(out_dir, "_smoke")
if (quick) model_dir <- file.path(out_dir, "models")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)

seed <- 20260905
set.seed(seed)
cfg_path <- file.path(ROOT, "registry", "ml_config.json")
cfg <- fromJSON(cfg_path, simplifyVector = FALSE)
if (!is.null(cfg$seed)) seed <- cfg$seed
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

read_input <- function() {
  p <- file.path(ROOT, "data", "ml", "recurrence.tsv")
  d <- read.delim(p, check.names = FALSE, stringsAsFactors = FALSE)
  stopifnot(nrow(d) == 87, sum(d$event) == 17, sum(d$recurrence_primary == 1) == 82)
  stopifnot(sum(d$event[d$recurrence_primary == 1]) == 14)
  stopifnot(sum(d$event == 1 & d$duration_days <= 1095 & d$recurrence_primary == 1) == 7)
  stopifnot(sum(d$event == 1 & d$duration_days <= 1825 & d$recurrence_primary == 1) == 11)
  d
}

make_folds <- function(event, k, seed0) {
  set.seed(seed0)
  fold <- integer(length(event))
  for (cls in c(0, 1)) {
    idx <- which(event == cls)
    idx <- sample(idx)
    fold[idx] <- rep(seq_len(k), length.out = length(idx))
  }
  fold
}

make_group_folds <- function(groups, event, k, seed0) {
  set.seed(seed0)
  u <- unique(groups)
  ge <- vapply(u, function(g) max(event[groups == g]), numeric(1))
  gf <- integer(length(u))
  for (cls in c(0, 1)) {
    idx <- which(ge == cls)
    idx <- sample(idx)
    gf[idx] <- rep(seq_len(k), length.out = length(idx))
  }
  setNames(gf, u)[as.character(groups)]
}

make_outer_folds <- function(d, repeats, k, seed0, tag) {
  rows <- list()
  for (r in seq_len(repeats)) {
    f <- make_folds(d$event, k, seed0 + 1000L * r)
    tab <- table(f, d$event)
    if (any(rowSums(tab) == 0) || any(tab[, "1"] == 0)) stop("outer fold without events")
    rows[[r]] <- data.frame(repeat_id = r, fold = f, study_id = d$study_id, event = d$event,
                            duration_days = d$duration_days, analysis = tag)
  }
  out <- do.call(rbind, rows)
  names(out)[names(out) == "repeat_id"] <- "repeat"
  out
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

q95_exact <- function(x) {
  x <- sort(as.numeric(x[is.finite(x)]))
  if (!length(x)) return(NA_real_)
  x[1 + ceiling((length(x) - 1) * 0.95)]
}

add_q95_columns <- function(d, train_idx, q_genes) {
  out <- d
  thresholds <- vapply(q_genes, function(g) q95_exact(d[train_idx, paste0("N_", g)]), numeric(1))
  for (g in q_genes) out[[paste0("Q95_", g)]] <- as.integer(out[[paste0("T_", g)]] > thresholds[g])
  attr(out, "q95_thresholds") <- thresholds
  out
}

model_blocks <- function(kind = "primary") {
  list(
    clinical = unlist(cfg$recurrence$clinical_features),
    tumor10 = paste0("T_", genes),
    combined = c(unlist(cfg$recurrence$clinical_features), paste0("T_", genes))
  )
}

cox_deviance <- function(time, event, eta) {
  y <- Surv(time, event)
  tryCatch(glmnet::coxnet.deviance(pred = eta, y = y), error = function(e) {
    warning("Invalid inner Cox deviance: ", conditionMessage(e), call. = FALSE)
    Inf
  })
}

select_glmnet_lambda <- function(x, time, event, alpha, lambdas, inner_k, seed0) {
  fold <- make_folds(event, inner_k, seed0)
  select_glmnet_lambda_with_folds(x, time, event, alpha, lambdas, fold)
}

predict_available_cox_path <- function(fit, newx, lambdas) {
  if (is.null(fit$jerr) || fit$jerr > 0) {
    stop("Unrecoverable glmnet path error; jerr=", fit$jerr)
  }
  matched <- match(lambdas, fit$lambda)
  available <- which(!is.na(matched))
  out <- matrix(NA_real_, nrow = nrow(newx), ncol = length(lambdas))
  if (!length(available)) stop("No fitted Cox penalty candidates")
  # A truncated path must never extrapolate an unavailable small penalty.
  predictions <- as.matrix(predict(fit, newx = newx, type = "link"))
  out[, available] <- predictions[, matched[available], drop = FALSE]
  if (length(available) != length(lambdas) || fit$jerr != 0) {
    warning("Truncated Cox path: ", length(available), "/", length(lambdas),
            " fitted candidates; unavailable candidates excluded from inner selection; jerr=", fit$jerr,
            call. = FALSE)
  }
  out
}

select_glmnet_lambda_with_folds <- function(x, time, event, alpha, lambdas, fold) {
  inner_k <- length(unique(fold))
  losses <- matrix(NA_real_, nrow = inner_k, ncol = length(lambdas))
  for (j in seq_len(inner_k)) {
    tr <- fold != j; va <- fold == j
    st <- scale_train_test(x[tr, , drop = FALSE], x[va, , drop = FALSE])
    fit <- glmnet(st$train, Surv(time[tr], event[tr]), family = "cox",
                  alpha = alpha, lambda = lambdas, standardize = FALSE)
    pred <- predict_available_cox_path(fit, st$test, lambdas)
    for (l in seq_along(lambdas)) if (all(is.finite(pred[, l])))
      losses[j, l] <- cox_deviance(time[va], event[va], pred[, l])
  }
  valid <- apply(losses, 2, function(v) all(is.finite(v)))
  if (!any(valid)) stop("no finite inner-CV Cox deviance candidate")
  mean_loss <- rep(NA_real_, length(lambdas))
  se_loss <- rep(NA_real_, length(lambdas))
  mean_loss[valid] <- colMeans(losses[, valid, drop = FALSE])
  se_loss[valid] <- apply(losses[, valid, drop = FALSE], 2, sd) / sqrt(inner_k)
  best <- which.min(mean_loss)
  eligible <- which(mean_loss <= mean_loss[best] + se_loss[best])
  chosen <- max(eligible)
  list(lambda = lambdas[chosen], best_lambda = lambdas[best], inner_fold = fold,
       tuning = data.frame(lambda = lambdas, mean_loss = mean_loss, se_loss = se_loss,
                           valid = valid, t(losses), check.names = FALSE))
}

fit_predict_glmnet <- function(d, train_idx, test_idx, features, alpha, repeat_id, fold_id, block, model) {
  x <- as.matrix(d[, features, drop = FALSE])
  time <- d$duration_days; event <- d$event
  lambdas <- lambda_grid()
  sel <- select_glmnet_lambda(x[train_idx, , drop = FALSE], time[train_idx], event[train_idx],
                              alpha, lambdas, cfg$recurrence$inner_folds,
                              inner_seed(repeat_id, fold_id))
  st <- scale_train_test(x[train_idx, , drop = FALSE], x[test_idx, , drop = FALSE])
  fit <- glmnet(st$train, Surv(time[train_idx], event[train_idx]), family = "cox",
                alpha = alpha, lambda = sel$lambda, standardize = FALSE)
    if (is.null(fit$jerr) || fit$jerr != 0) stop("glmnet path did not converge; jerr=", fit$jerr)
  eta_train_raw <- as.numeric(predict(fit, newx = st$train, s = sel$lambda, type = "link"))
  center_eta <- mean(eta_train_raw)
  eta_test <- as.numeric(predict(fit, newx = st$test, s = sel$lambda, type = "link")) - center_eta
  eta_train <- eta_train_raw - center_eta
  bh <- breslow_baseline(time[train_idx], event[train_idx], eta_train, center = 0)
  co <- as.numeric(coef(fit, s = sel$lambda))
  names(co) <- features
  risk1095 <- risk_from_cox_baseline(bh, eta_test, 1095)
  risk1825 <- risk_from_cox_baseline(bh, eta_test, 1825)
  risk_grid <- risk_grid_from_cox(bh, eta_test, time_grid)
  list(pred = eta_test, risk1095 = risk1095, risk1825 = risk1825,
       risk_grid = risk_grid,
       fit = list(model_object = fit, lambda = sel$lambda, best_lambda = sel$best_lambda,
                  coef = co, center = st$center, scale = st$scale,
                  eta_center = center_eta, baseline_hazard = bh, inner_fold = sel$inner_fold),
       tuning = data.frame(sel$tuning, repeat_id = repeat_id, fold = fold_id, block = block, model = model))
}

fit_predict_q95 <- function(d, train_idx, test_idx, q_genes, base_features, repeat_id, fold_id, block) {
  q_features <- paste0("Q95_", q_genes)
  features <- c(base_features, q_features)
  lambdas <- lambda_grid()
  outer_train <- d[train_idx, , drop = FALSE]
  inner_fold <- make_folds(outer_train$event, cfg$recurrence$inner_folds,
                           inner_seed(repeat_id, fold_id))
  losses <- matrix(NA_real_, nrow = cfg$recurrence$inner_folds, ncol = length(lambdas))
  for (j in seq_len(cfg$recurrence$inner_folds)) {
    inner_tr_local <- which(inner_fold != j)
    inner_va_local <- which(inner_fold == j)
    inner_tr <- train_idx[inner_tr_local]
    inner_va <- train_idx[inner_va_local]
    dq <- add_q95_columns(d, inner_tr, q_genes)
    xtr <- as.matrix(dq[inner_tr, features, drop = FALSE])
    xva <- as.matrix(dq[inner_va, features, drop = FALSE])
    st <- scale_train_test(xtr, xva)
    fit <- glmnet(st$train, Surv(d$duration_days[inner_tr], d$event[inner_tr]),
                  family = "cox", alpha = 0, lambda = lambdas, standardize = FALSE)
    pred <- predict_available_cox_path(fit, st$test, lambdas)
    for (l in seq_along(lambdas)) if (all(is.finite(pred[, l])))
      losses[j, l] <- cox_deviance(d$duration_days[inner_va], d$event[inner_va], pred[, l])
  }
  valid <- apply(losses, 2, function(v) all(is.finite(v)))
  if (!any(valid)) stop("no finite inner-CV Q95 Cox deviance candidate")
  mean_loss <- rep(NA_real_, length(lambdas)); se_loss <- rep(NA_real_, length(lambdas))
  mean_loss[valid] <- colMeans(losses[, valid, drop = FALSE])
  se_loss[valid] <- apply(losses[, valid, drop = FALSE], 2, sd) / sqrt(cfg$recurrence$inner_folds)
  best <- which.min(mean_loss)
  chosen <- max(which(mean_loss <= mean_loss[best] + se_loss[best]))
  dq <- add_q95_columns(d, train_idx, q_genes)
  thresholds <- attr(dq, "q95_thresholds")
  xtr <- as.matrix(dq[train_idx, features, drop = FALSE])
  xte <- as.matrix(dq[test_idx, features, drop = FALSE])
  st <- scale_train_test(xtr, xte)
  fit <- glmnet(st$train, Surv(d$duration_days[train_idx], d$event[train_idx]),
                family = "cox", alpha = 0, lambda = lambdas[chosen], standardize = FALSE)
    if (is.null(fit$jerr) || fit$jerr != 0) stop("glmnet path did not converge; jerr=", fit$jerr)
  eta_train_raw <- as.numeric(predict(fit, newx = st$train, s = lambdas[chosen], type = "link"))
  center_eta <- mean(eta_train_raw)
  eta_test <- as.numeric(predict(fit, newx = st$test, s = lambdas[chosen], type = "link")) - center_eta
  bh <- breslow_baseline(d$duration_days[train_idx], d$event[train_idx],
                         eta_train_raw - center_eta, center = 0)
  co <- as.numeric(coef(fit, s = lambdas[chosen])); names(co) <- features
  list(pred = eta_test,
       risk1095 = risk_from_cox_baseline(bh, eta_test, 1095),
       risk1825 = risk_from_cox_baseline(bh, eta_test, 1825),
       risk_grid = risk_grid_from_cox(bh, eta_test, time_grid),
       fit = list(model_object = fit, lambda = lambdas[chosen], best_lambda = lambdas[best],
                  coef = co, center = st$center, scale = st$scale, eta_center = center_eta,
                  baseline_hazard = bh, inner_fold = inner_fold, q95_thresholds = thresholds),
       tuning = data.frame(lambda = lambdas, mean_loss = mean_loss, se_loss = se_loss,
                           valid = valid, t(losses), check.names = FALSE, repeat_id = repeat_id, fold = fold_id,
                           block = block, model = "ridge"))
}

fit_ridge_full <- function(d, features, block, inner_groups = NULL, seed0 = seed) {
  x <- as.matrix(d[, features, drop = FALSE])
  folds <- if (is.null(inner_groups)) make_folds(d$event, cfg$recurrence$inner_folds, seed0) else
    make_group_folds(inner_groups, d$event, cfg$recurrence$inner_folds, seed0)
  lambdas <- lambda_grid()
  sel <- select_glmnet_lambda_with_folds(x, d$duration_days, d$event, 0, lambdas, folds)
  st <- scale_train_test(x, x)
  fit <- glmnet(st$train, Surv(d$duration_days, d$event), family = "cox",
                alpha = 0, lambda = sel$lambda, standardize = FALSE)
    if (is.null(fit$jerr) || fit$jerr != 0) stop("glmnet path did not converge; jerr=", fit$jerr)
  eta_raw <- as.numeric(predict(fit, newx = st$train, s = sel$lambda, type = "link"))
  center_eta <- mean(eta_raw)
  bh <- breslow_baseline(d$duration_days, d$event, eta_raw - center_eta, center = 0)
  list(fit = fit, lambda = sel$lambda, center = st$center, scale = st$scale,
       eta_center = center_eta, baseline = bh, features = features, block = block,
       tuning = transform(sel$tuning, block = block, model = "ridge"),
       inner_fold = folds)
}

predict_ridge_full <- function(obj, d) {
  x <- as.matrix(d[, obj$features, drop = FALSE])
  xs <- sweep(sweep(x, 2, obj$center, "-"), 2, obj$scale, "/")
  eta <- as.numeric(predict(obj$fit, newx = xs, s = obj$lambda, type = "link")) - obj$eta_center
  list(score = eta,
       risk1095 = cox_risk(obj$baseline, eta, 1095, center = 0),
       risk1825 = cox_risk(obj$baseline, eta, 1825, center = 0),
       risk_grid = sapply(time_grid, function(tt) cox_risk(obj$baseline, eta, tt, center = 0)))
}

single_eval_oof <- function(d, pred, block, analysis, km_source = d) {
  gkm <- km_censor(km_source$duration_days, km_source$event)
  g1095 <- vapply(d$duration_days * 0 + 1095, function(tt) g_at(gkm, tt), numeric(1))
  g1825 <- vapply(d$duration_days * 0 + 1825, function(tt) g_at(gkm, tt), numeric(1))
  gtime <- vapply(d$duration_days, function(tt) g_before(gkm, tt), numeric(1))
  ggrid <- matrix(vapply(time_grid, function(tt) g_at(gkm, tt), numeric(1)),
                  nrow = nrow(d), ncol = length(time_grid), byrow = TRUE)
  colnames(ggrid) <- g_grid_cols
  rg <- pred$risk_grid
  if (is.null(dim(rg))) rg <- matrix(rg, nrow = nrow(d))
  colnames(rg) <- grid_cols
  out <- cbind(data.frame(analysis = analysis, repeat_id = 1, fold = 1, study_id = d$study_id,
                          event = d$event, duration_days = d$duration_days, block = block,
                          model = "ridge", risk_score = pred$score, risk_1095 = pred$risk1095,
                          risk_1825 = pred$risk1825, G_1095 = g1095, G_1825 = g1825,
                          G_time = gtime),
               as.data.frame(rg, check.names = FALSE),
               as.data.frame(ggrid, check.names = FALSE))
  names(out)[names(out) == "repeat_id"] <- "repeat"
  out
}

event_km_survival_at <- function(sf_time, sf_surv, t) {
  if (!length(sf_time) || t < min(sf_time)) return(1)
  idx <- max(which(sf_time <= t))
  sf_surv[idx]
}

write_null_km_outputs <- function(d, folds, prefix = "primary") {
  rows <- list(); ix <- 1
  for (r in sort(unique(folds[["repeat"]]))) {
    for (f in sort(unique(folds$fold))) {
      te <- which(d$study_id %in% folds$study_id[folds[["repeat"]] == r & folds$fold == f])
      tr <- setdiff(seq_len(nrow(d)), te)
      sf <- survfit(Surv(d$duration_days[tr], d$event[tr]) ~ 1)
      risk_grid <- matrix(vapply(time_grid, function(tt) 1 - event_km_survival_at(sf$time, sf$surv, tt), numeric(1)),
                          nrow = length(te), ncol = length(time_grid), byrow = TRUE)
      colnames(risk_grid) <- grid_cols
      risk1095 <- rep(1 - event_km_survival_at(sf$time, sf$surv, 1095), length(te))
      risk1825 <- rep(1 - event_km_survival_at(sf$time, sf$surv, 1825), length(te))
      gkm <- km_censor(d$duration_days[tr], d$event[tr])
      g1095 <- vapply(d$duration_days[te] * 0 + 1095, function(tt) g_at(gkm, tt), numeric(1))
      g1825 <- vapply(d$duration_days[te] * 0 + 1825, function(tt) g_at(gkm, tt), numeric(1))
      gtime <- vapply(d$duration_days[te], function(tt) g_before(gkm, tt), numeric(1))
      ggrid <- matrix(vapply(time_grid, function(tt) g_at(gkm, tt), numeric(1)),
                      nrow = length(te), ncol = length(time_grid), byrow = TRUE)
      colnames(ggrid) <- g_grid_cols
      rows[[ix]] <- cbind(data.frame(analysis = paste0(prefix, "_null_km"), repeat_id = r,
                                     fold = f, study_id = d$study_id[te],
                                     event = d$event[te], duration_days = d$duration_days[te],
                                     block = "null_km", model = "km",
                                     risk_score = risk1825,
                                     risk_1095 = risk1095, risk_1825 = risk1825,
                                     G_1095 = g1095, G_1825 = g1825, G_time = gtime),
                           as.data.frame(risk_grid, check.names = FALSE),
                           as.data.frame(ggrid, check.names = FALSE))
      ix <- ix + 1
    }
  }
  oof <- do.call(rbind, rows); names(oof)[names(oof) == "repeat_id"] <- "repeat"
  repmet <- repeat_metrics(oof)
  agg <- aggregate_metrics(repmet)
  write.csv(oof, file.path(out_dir, paste0(prefix, "_null_km_oof_predictions.csv")), row.names = FALSE)
  write.csv(repmet, file.path(out_dir, paste0(prefix, "_null_km_repeat_metrics.csv")), row.names = FALSE)
  write.csv(agg, file.path(out_dir, paste0(prefix, "_null_km_metrics.csv")), row.names = FALSE)
  invisible(list(oof = oof, repeat_metrics = repmet, aggregate = agg))
}

harrell_c <- function(time, event, score) {
  ok <- is.finite(score)
  time <- time[ok]; event <- event[ok]; score <- score[ok]
  num <- den <- 0
  n <- length(time)
  if (n < 2) return(NA_real_)
  for (i in seq_len(n - 1)) for (j in (i + 1):n) {
    if (time[i] == time[j]) next
    if (time[i] < time[j] && event[i] == 1) {
      den <- den + 1; num <- num + (score[i] > score[j]) + 0.5 * (score[i] == score[j])
    } else if (time[j] < time[i] && event[j] == 1) {
      den <- den + 1; num <- num + (score[j] > score[i]) + 0.5 * (score[i] == score[j])
    }
  }
  if (den == 0) NA_real_ else num / den
}

select_rsf <- function(x, time, event, inner_k, seed0) {
  fold <- make_folds(event, inner_k, seed0)
  p <- ncol(x)
  grid <- expand.grid(node = as.integer(unlist(cfg$recurrence$forest_node_size)),
                      mtry_kind = unlist(cfg$recurrence$forest_mtry), stringsAsFactors = FALSE)
  grid$mtry <- ifelse(grid$mtry_kind == "sqrt", max(1L, floor(sqrt(p))), p)
  score <- numeric(nrow(grid))
  fold_scores <- matrix(NA_real_, nrow = nrow(grid), ncol = inner_k)
  for (g in seq_len(nrow(grid))) {
    vals <- numeric(inner_k)
    for (j in seq_len(inner_k)) {
      tr <- fold != j; va <- fold == j
      dat_tr <- data.frame(time = time[tr], event = event[tr], x[tr, , drop = FALSE])
      dat_va <- data.frame(x[va, , drop = FALSE])
      fit <- ranger(Surv(time, event) ~ ., data = dat_tr, num.trees = as.integer(cfg$recurrence$forest_trees),
                    mtry = grid$mtry[g], min.node.size = grid$node[g], max.depth = cfg$recurrence$forest_depth,
                    splitrule = "logrank", respect.unordered.factors = "ignore",
                    num.threads = 1, seed = seed0 + g * 19 + j)
      pr <- predict(fit, data = dat_va)$chf
      risk <- rowSums(pr)
      vals[j] <- harrell_c(time[va], event[va], risk)
    }
    fold_scores[g, ] <- vals
    score[g] <- if (all(is.finite(vals))) mean(vals) else NA_real_
  }
  if (!any(is.finite(score))) stop("no finite inner-CV RSF candidate")
  best_score <- max(score[is.finite(score)])
  eligible <- which(score >= best_score - 1e-12)
  # Deterministic simpler tie: larger node size, then sqrt mtry.
  tie_order <- order(-grid$node[eligible], grid$mtry[eligible])
  chosen <- eligible[tie_order[1]]
  list(node = grid$node[chosen], mtry = grid$mtry[chosen], mtry_kind = grid$mtry_kind[chosen],
       inner_fold = fold,
       tuning = data.frame(node = grid$node, mtry = grid$mtry, mtry_kind = grid$mtry_kind, mean_c = score,
                           valid = is.finite(score), fold_scores, check.names = FALSE))
}

fit_predict_rsf <- function(d, train_idx, test_idx, features, repeat_id, fold_id, block) {
  x <- as.matrix(d[, features, drop = FALSE])
  time <- d$duration_days; event <- d$event
  st <- scale_train_test(x[train_idx, , drop = FALSE], x[test_idx, , drop = FALSE])
  colnames(st$train) <- features; colnames(st$test) <- features
  sel <- select_rsf(x[train_idx, , drop = FALSE], time[train_idx], event[train_idx], cfg$recurrence$inner_folds,
                    inner_seed(repeat_id, fold_id))
  dat_tr <- data.frame(time = time[train_idx], event = event[train_idx], st$train)
  fit <- ranger(Surv(time, event) ~ ., data = dat_tr, num.trees = as.integer(cfg$recurrence$forest_trees),
                mtry = sel$mtry, min.node.size = sel$node, max.depth = cfg$recurrence$forest_depth,
                splitrule = "logrank", respect.unordered.factors = "ignore",
                num.threads = 1,
                seed = seed + 1009L * repeat_id + fold_id)
  pr <- predict(fit, data = data.frame(st$test))
  risk <- rowSums(pr$chf)
  list(pred = risk, risk1095 = surv_at_ranger(pr, 1095), risk1825 = surv_at_ranger(pr, 1825),
       risk_grid = risk_grid_from_ranger(pr, time_grid),
       fit = list(model_object = fit, node = sel$node, mtry = sel$mtry, mtry_kind = sel$mtry_kind,
                  center = st$center, scale = st$scale, inner_fold = sel$inner_fold),
       tuning = data.frame(sel$tuning, repeat_id = repeat_id, fold = fold_id, block = block, model = "survival_forest"))
}

km_censor <- function(time, event) censor_km(time, event)

base_hazard_at <- function(bh, t) {
  if (nrow(bh) == 0 || t < min(bh$time)) return(0)
  idx <- max(which(bh$time <= t))
  bh$hazard[idx]
}

risk_from_cox_baseline <- function(bh, eta, tau) {
  h <- base_hazard_at(bh, tau)
  1 - exp(-h * exp(eta))
}

risk_grid_from_cox <- function(bh, eta, grid) {
  mat <- sapply(grid, function(tt) risk_from_cox_baseline(bh, eta, tt))
  colnames(mat) <- paste0("risk_grid_", seq_along(grid))
  mat
}

surv_at_ranger <- function(pred, tau) {
  times <- pred$unique.death.times
  eligible <- which(times <= tau)
  if (!length(eligible)) return(rep(0, nrow(pred$survival)))
  idx <- max(eligible)
  1 - pred$survival[, idx]
}

risk_grid_from_ranger <- function(pred, grid) {
  mat <- sapply(grid, function(tt) surv_at_ranger(pred, tt))
  colnames(mat) <- paste0("risk_grid_", seq_along(grid))
  mat
}

metrics_module <- file.path(ROOT, "ml", "recurrence_metrics.R")
if (!file.exists(metrics_module)) stop("required metrics module missing: ml/recurrence_metrics.R")
source(metrics_module, local = .GlobalEnv)

repeat_metrics <- function(oof) {
  out <- repeat_metrics_from_oof(oof, horizons = c(1095, 1825))
  out$ibs <- NA_real_
  for (r in sort(unique(oof[["repeat"]]))) {
    for (block in sort(unique(oof$block))) {
      for (model in sort(unique(oof$model))) {
        z <- oof[oof[["repeat"]] == r & oof$block == block & oof$model == model, , drop = FALSE]
        val <- ibs(z, max_horizon = 1825, n_grid = length(time_grid), risk_prefix = "risk_grid_")
        out$ibs[out[["repeat"]] == r & out$block == block & out$model == model] <- val
      }
    }
  }
  out
}

aggregate_metrics <- function(repmet) {
  out <- aggregate_repeat_metrics(repmet)
  ib <- do.call(rbind, by(repmet, repmet[c("block", "model", "horizon_days")], function(z) {
    data.frame(block = z$block[1], model = z$model[1], horizon_days = z$horizon_days[1],
               ibs_mean = if (all(is.finite(z$ibs))) mean(z$ibs) else NA_real_,
               ibs_n_estimable = sum(is.finite(z$ibs)),
               ibs_status = if (all(is.finite(z$ibs))) "estimable" else "nonestimable", stringsAsFactors = FALSE)
  }))
  merge(out, ib, by = c("block", "model", "horizon_days"), all.x = TRUE, sort = FALSE)
}

source_fingerprints <- function() {
  paths <- c(input = file.path(ROOT, "data/ml/recurrence.tsv"), config = cfg_path,
             runner = file.path(ROOT, "scripts/run_recurrence_ml.R"),
             metrics = metrics_module, resampling = file.path(ROOT, "ml/recurrence_resampling.R"))
  vapply(paths, function(p) if (file.exists(p)) sha256_file(p) else "missing", character(1))
}

atomic_save_rds <- function(value, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  tmp <- paste0(path, ".", Sys.getpid(), ".tmp")
  on.exit(unlink(tmp), add = TRUE)
  saveRDS(value, tmp)
  if (!file.rename(tmp, path)) stop("Could not atomically write ", path)
  invisible(path)
}

run_nested <- function(d, analysis_name = "primary", repeats = cfg$recurrence$outer_repeats,
                       models = cfg$recurrence$models, blocks_override = NULL,
                       n_workers = min(3L, as.integer(Sys.getenv("ML_RECURRENCE_WORKERS", "2"))),
                       save_models = TRUE, checkpoint = save_models, metric_only = FALSE) {
  blocks <- if (is.null(blocks_override)) model_blocks() else blocks_override
  models <- unlist(models)
  folds <- make_outer_folds(d, repeats, cfg$recurrence$outer_folds, seed, analysis_name)
  fingerprints <- source_fingerprints()
  key <- sha256_text(toJSON(list(source = fingerprints, data = d, blocks = blocks,
                               models = models, repeats = repeats, analysis = analysis_name,
                               save_models = save_models, metric_only = metric_only),
                          dataframe = "columns", digits = 17, auto_unbox = TRUE, na = "null"))
  cache_dir <- file.path(out_dir, ".cache", key)
  current_model_dir <- file.path(model_dir, key)
  if (checkpoint) dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
  if (save_models) dir.create(current_model_dir, recursive = TRUE, showWarnings = FALSE)
  jobs <- expand.grid(model = models, block = names(blocks),
                      fold = seq_len(cfg$recurrence$outer_folds), repeat_id = seq_len(repeats),
                      stringsAsFactors = FALSE)
  one_job <- function(j) {
    r <- jobs$repeat_id[j]; f <- jobs$fold[j]
    block <- jobs$block[j]; model <- jobs$model[j]; features <- blocks[[block]]
    stem <- sprintf("%s_repeat%03d_fold%02d_%s_%s", analysis_name, r, f, block, model)
    cache_path <- file.path(cache_dir, paste0(stem, ".rds"))
    if (checkpoint && file.exists(cache_path)) {
      previous <- tryCatch(readRDS(cache_path), error = function(e) NULL)
      if (!is.null(previous) && identical(previous$key, key) &&
          (!save_models || (file.exists(previous$model_file) &&
                           identical(sha256_file(previous$model_file), previous$model_sha256)))) return(previous$result)
    }
    warnings <- character(0)
    result <- tryCatch(withCallingHandlers({
      test_ids <- folds$study_id[folds[["repeat"]] == r & folds$fold == f]
      te <- which(d$study_id %in% test_ids); tr <- setdiff(seq_len(nrow(d)), te)
      stopifnot(!length(intersect(d$study_id[tr], d$study_id[te])))
      q_genes <- sub("^Q95_", "", features[startsWith(features, "Q95_")])
      if (length(q_genes)) {
        if (model != "ridge") stop("Q95 sensitivity is ridge-only")
        fp <- fit_predict_q95(d, tr, te, q_genes, features[!startsWith(features, "Q95_")], r, f, block)
      } else if (model == "survival_forest") {
        fp <- fit_predict_rsf(d, tr, te, features, r, f, block)
      } else {
        alpha <- if (model == "ridge") 0 else cfg$recurrence$elastic_alpha
        fp <- fit_predict_glmnet(d, tr, te, features, alpha, r, f, block, model)
      }
      if (length(fp$pred) != length(te) || any(!is.finite(fp$pred))) stop("Invalid held-out risk score")
      valid_probability <- function(x) length(x) == length(te) && all(is.finite(x)) && all(x >= 0 & x <= 1)
      if (!valid_probability(fp$risk1095) || !valid_probability(fp$risk1825)) stop("Invalid held-out horizon probability")
      if (!is.matrix(fp$risk_grid) || !identical(dim(fp$risk_grid), c(length(te), length(time_grid))) ||
          any(!is.finite(fp$risk_grid)) || any(fp$risk_grid < 0 | fp$risk_grid > 1)) stop("Invalid held-out probability grid")
      gkm <- km_censor(d$duration_days[tr], d$event[tr])
      gtime <- vapply(d$duration_days[te], function(tt) g_before(gkm, tt), numeric(1))
      ggrid <- matrix(vapply(time_grid, function(tt) g_at(gkm, tt), numeric(1)),
                      nrow = length(te), ncol = length(time_grid), byrow = TRUE)
      colnames(ggrid) <- g_grid_cols
      oof <- data.frame(analysis = analysis_name, repeat_id = r, fold = f,
                        study_id = d$study_id[te], event = d$event[te], duration_days = d$duration_days[te],
                        block = block, model = model, risk_score = fp$pred,
                        risk_1095 = fp$risk1095, risk_1825 = fp$risk1825,
                        G_1095 = g_at(gkm, 1095), G_1825 = g_at(gkm, 1825), G_time = gtime)
      if (!metric_only) oof <- cbind(oof, as.data.frame(fp$risk_grid, check.names = FALSE),
                                    as.data.frame(ggrid, check.names = FALSE))
      names(oof)[names(oof) == "repeat_id"] <- "repeat"
      artifact <- list(analysis = analysis_name, repeat_id = r, fold = f, block = block, model = model,
                       features = features, source_hashes = fingerprints, run_key = key,
                       train_study_ids = d$study_id[tr], test_study_ids = d$study_id[te],
                       inner_study_ids = d$study_id[tr], fit = fp$fit, censoring_km = gkm,
                       warnings = warnings)
      model_file <- file.path(current_model_dir, paste0(stem, ".rds"))
      if (save_models) atomic_save_rds(artifact, model_file)
      model_summary <- data.frame(analysis = analysis_name, repeat_id = r, fold = f, block = block,
                                  model = model, selected_lambda = fp$fit$lambda %||% NA_real_,
                                  model_path = if (save_models) sub(paste0("^", ROOT, "/"), "", model_file) else NA_character_,
                                  features = paste(features, collapse = ";"), status = "complete",
                                  warning_count = length(warnings), warnings = paste(unique(warnings), collapse = " | "),
                                  stringsAsFactors = FALSE)
      names(model_summary)[names(model_summary) == "repeat_id"] <- "repeat"
      tuning <- fp$tuning
      names(tuning)[names(tuning) == "repeat_id"] <- "repeat"
      list(oof = oof, tuning = tuning, model_summary = model_summary)
    }, warning = function(w) { warnings <<- c(warnings, conditionMessage(w)); invokeRestart("muffleWarning") }),
    error = function(e) {
      dir.create(file.path(out_dir, "failures"), recursive = TRUE, showWarnings = FALSE)
      write(toJSON(list(job = stem, error = conditionMessage(e), warnings = warnings,
                        key = key, source = fingerprints), auto_unbox = TRUE, pretty = TRUE),
            file.path(out_dir, "failures", paste0(stem, ".json")))
      stop(stem, ": ", conditionMessage(e), call. = FALSE)
    })
    if (checkpoint) {
      model_file <- file.path(current_model_dir, paste0(stem, ".rds"))
      atomic_save_rds(list(key = key, model_file = model_file,
                          model_sha256 = if (save_models) sha256_file(model_file) else NA_character_,
                          result = result), cache_path)
    }
    result
  }
  if (Sys.getenv("ML_RECURRENCE_IN_RESAMPLE", "0") == "1") n_workers <- 1L
  n_workers <- max(1L, min(3L, as.integer(n_workers), nrow(jobs)))
  fits <- if (n_workers > 1L) parallel::mclapply(seq_len(nrow(jobs)), one_job,
                                                mc.cores = n_workers, mc.preschedule = FALSE, mc.set.seed = FALSE) else
    lapply(seq_len(nrow(jobs)), one_job)
  failed <- vapply(fits, inherits, logical(1), "try-error")
  if (any(failed)) stop("Nested fit stage failed; see preserved job failure records: ", paste(which(failed), collapse = ","))
  if (!identical(fingerprints, source_fingerprints())) stop("Source/input changed during nested run")
  oof <- do.call(rbind, lapply(fits, `[[`, "oof"))
  tuning <- bind_rows_fill(lapply(fits, `[[`, "tuning"))
  model_summary <- do.call(rbind, lapply(fits, `[[`, "model_summary"))
  if (metric_only) {
    reps <- lapply(split(oof, oof[["repeat"]]), function(z) {
      uc <- uno_counts_by_fold(z, 1825)
      data.frame(repeat_id = z[["repeat"]][1], block = z$block[1], model = z$model[1],
                  horizon_days = 1825, uno_c = uc[["estimate"]], uno_num = uc[["numerator"]],
                  uno_den = uc[["denominator"]])
    })
    repmet <- do.call(rbind, reps); names(repmet)[1] <- "repeat"
    aggregate <- data.frame(block = repmet$block[1], model = repmet$model[1], horizon_days = 1825,
                            n_repeats = nrow(repmet),
                            uno_c_mean = if (all(is.finite(repmet$uno_c))) mean(repmet$uno_c) else NA_real_)
  } else {
    repmet <- repeat_metrics(oof)
    aggregate <- aggregate_metrics(repmet)
  }
  message(sprintf("[%s] %d outer fits complete (%d repeats)", analysis_name, nrow(jobs), repeats))
  list(folds = folds, oof = oof, tuning = tuning, model_summary = model_summary,
       repeat_metrics = repmet, aggregate = aggregate, source_hashes = fingerprints, run_key = key)
}

run_q95_sensitivity <- function(d, repeats = 25, q_genes = genes, base_features = character(0),
                                block_name = "q95_binary_train_normals") {
  d <- d[d$recurrence_primary == 1, , drop = FALSE]
  analysis_name <- paste0("sensitivity_", block_name)
  block <- setNames(list(c(base_features, paste0("Q95_", q_genes))), block_name)
  result <- run_nested(d, analysis_name, repeats, models = "ridge", blocks_override = block)
  write_outputs(analysis_name, result)
  transform(result$aggregate, sensitivity = block_name)
}

run_sensitivity <- function(d, repeats = 5) {
  variants <- list(
    all87_stage_advanced = transform(d, stage_iii = stage_advanced),
    no_stage = d,
    cea_binary = d,
    delta10 = d,
    clinical_delta10 = d
  )
  blocks <- list(
    all87_stage_advanced = c("stage_advanced", "cea_log1p", "lvi", paste0("T_", genes)),
    no_stage = c("cea_log1p", "lvi", paste0("T_", genes)),
    cea_binary = c("stage_iii", "cea_binary", "lvi", paste0("T_", genes)),
    delta10 = paste0("D_", genes),
    clinical_delta10 = c("stage_iii", "cea_log1p", "lvi", paste0("D_", genes))
  )
  rows <- list(); ix <- 1
  for (nm in names(variants)) {
    dv <- if (startsWith(nm, "all87")) variants[[nm]] else variants[[nm]][variants[[nm]]$recurrence_primary == 1, ]
    res <- run_nested(dv, paste0("sensitivity_", nm), repeats = repeats, models = c("ridge"),
                      blocks_override = setNames(list(blocks[[nm]]), nm))
    write_outputs(paste0("sensitivity_", nm), res)
    rows[[ix]] <- transform(subset(res$aggregate, model == "ridge"), sensitivity = nm); ix <- ix + 1
  }
  rows[[ix]] <- run_q95_sensitivity(d, repeats = repeats); ix <- ix + 1
  rows[[ix]] <- run_q95_sensitivity(d, repeats = repeats, q_genes = c("RALYL", "SFMBT2"),
                                    base_features = "cea_binary",
                                    block_name = "historical_cea_q95_ralyl_sfmbt2")
  do.call(rbind, rows)
}

write_outputs <- function(prefix, res) {
  write.csv(res$folds, file.path(out_dir, paste0(prefix, "_folds.csv")), row.names = FALSE)
  write.csv(res$oof, file.path(out_dir, paste0(prefix, "_oof_predictions.csv")), row.names = FALSE)
  write.csv(res$tuning, file.path(out_dir, paste0(prefix, "_inner_tuning.csv")), row.names = FALSE)
  write.csv(res$model_summary, file.path(out_dir, paste0(prefix, "_selected_hyperparameters.csv")), row.names = FALSE)
  write.csv(res$repeat_metrics, file.path(out_dir, paste0(prefix, "_repeat_metrics.csv")), row.names = FALSE)
  write.csv(res$aggregate, file.path(out_dir, paste0(prefix, "_metrics.csv")), row.names = FALSE)
}

read_primary_oof <- function() {
  p <- file.path(out_dir, "primary_oof_predictions.csv")
  if (!file.exists(p)) stop("primary_oof_predictions.csv is required for this stage")
  read.csv(p, check.names = FALSE)
}

read_primary_repeat_metrics <- function() {
  p <- file.path(out_dir, "primary_repeat_metrics.csv")
  if (!file.exists(p)) stop("primary_repeat_metrics.csv is required for this stage")
  read.csv(p, check.names = FALSE)
}

write_stage_manifest <- function(stage_name, expected_fingerprints = source_fingerprints()) {
  if (!identical(expected_fingerprints, source_fingerprints())) stop("Source/input changed during stage")
  files <- list.files(out_dir, pattern = "\\.(csv|json)$", full.names = TRUE)
  files <- files[!startsWith(basename(files), "manifest_")]
  payload <- list(
    stage = stage_name,
    generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
    quick = quick,
    seed = seed,
    hashes = as.list(expected_fingerprints),
    outputs = lapply(files, function(p) list(file = sub(paste0("^", ROOT, "/?"), "", p),
                                             sha256 = sha256_file(p)))
  )
  write(toJSON(payload, pretty = TRUE, auto_unbox = TRUE),
        file.path(out_dir, paste0("manifest_", stage_name, ".json")))
}

run_self_tests <- function() {
  d <- read_input()
  f <- make_outer_folds(d[d$recurrence_primary == 1, ], 3, 4, seed, "test")
  stopifnot(all(table(f[["repeat"]], f$fold) >= 20), all(tapply(f$event, list(f[["repeat"]], f$fold), sum) >= 1))
  x <- matrix(rnorm(40), ncol = 2)
  st <- scale_train_test(x[1:10, ], x[11:20, ])
  st2 <- scale_train_test(x[1:10, ], x[11:20, ] + 1000)
  stopifnot(all.equal(st$center, st2$center), all.equal(st$scale, st2$scale))
  c0 <- harrell_c(c(1, 2, 3), c(1, 0, 0), c(1, 1, 1))
  stopifnot(abs(c0 - 0.5) < 1e-12)
  uc <- uno_counts_fold(c(1, 2, 3, 4), c(1, 0, 1, 0), c(4, 3, 2, 1), 5, rep(1, 4))
  stopifnot(is.finite(uc["estimate"]), uc["denominator"] > 0)
  invisible(TRUE)
}

resampling_module <- file.path(ROOT, "ml", "recurrence_resampling.R")
if (file.exists(resampling_module)) source(resampling_module, local = .GlobalEnv)

main <- function() {
  expected_fingerprints <- source_fingerprints()
  if (self_test) {
    run_self_tests()
    cat("recurrence_ml self-tests passed\n")
    return(invisible(TRUE))
  }
  d0 <- read_input()
  primary <- d0[d0$recurrence_primary == 1, ]
  reps <- if (quick) 2 else cfg$recurrence$outer_repeats
  boot_n <- if (quick) 10 else cfg$uncertainty$patient_bootstrap
  perm_n <- if (quick) 1 else cfg$uncertainty$permutations
  opt_n <- if (quick) 1 else cfg$uncertainty$optimism_bootstrap
  primary_res <- NULL
  ci <- NULL
  perm <- NULL
  if (stage %in% c("all", "primary")) {
    primary_res <- run_nested(primary, "primary", repeats = reps, models = unlist(cfg$recurrence$models))
    write_outputs("primary", primary_res)
    write_null_km_outputs(primary, primary_res$folds, "primary")
    write_stage_manifest("primary", expected_fingerprints)
  }
  if (stage %in% c("all", "bootstrap")) {
    oof_for_ci <- if (!is.null(primary_res)) primary_res$oof else read_primary_oof()
    boot <- fast_bootstrap_ci(oof_for_ci,
                              metrics = c("uno_c", "auc", "brier", "harrell_c", "ibs"),
                              B = boot_n, seed = seed + 404,
                              horizons = c(1095, 1825),
                              include_deltas = TRUE)
    ci <- boot$ci
    write.csv(boot$ci, file.path(out_dir, "primary_bootstrap_ci.csv"), row.names = FALSE)
    write.csv(boot$draws, file.path(out_dir, "primary_bootstrap_draws.csv"), row.names = FALSE)
    write.csv(boot$repeat_mean_draws, file.path(out_dir, "primary_bootstrap_repeat_mean_draws.csv"), row.names = FALSE)
    saveRDS(list(weights = boot$W, study_id = sort(unique(oof_for_ci$study_id)),
                 seed = seed + 404, source_hashes = expected_fingerprints),
            file.path(out_dir, "primary_bootstrap_patient_weights.rds"))
    write_stage_manifest("bootstrap", expected_fingerprints)
  }
  if (stage %in% c("all", "sensitivity")) {
    sens <- run_sensitivity(d0, repeats = if (quick) 1 else cfg$recurrence$outer_repeats)
    write.csv(sens, file.path(out_dir, "ridge_sensitivity_metrics.csv"), row.names = FALSE)
    write_stage_manifest("sensitivity", expected_fingerprints)
  }
  if (stage %in% c("all", "optimism")) {
    opt <- run_optimism(primary, B = opt_n)
    write.csv(opt, file.path(out_dir, "primary_ridge_optimism_bootstrap.csv"), row.names = FALSE)
    write_stage_manifest("optimism", expected_fingerprints)
  }
  if (stage %in% c("all", "permutation")) {
    perm <- run_permutation(primary, B = perm_n)
    write.csv(perm, file.path(out_dir, "primary_ridge_permutation.csv"), row.names = FALSE)
    write_stage_manifest("permutation", expected_fingerprints)
  }
  aggregate_primary <- if (!is.null(primary_res)) primary_res$aggregate else if (file.exists(file.path(out_dir, "primary_metrics.csv"))) read.csv(file.path(out_dir, "primary_metrics.csv")) else NULL
  if (is.null(ci) && file.exists(file.path(out_dir, "primary_bootstrap_ci.csv"))) ci <- read.csv(file.path(out_dir, "primary_bootstrap_ci.csv"))
  if (is.null(perm) && file.exists(file.path(out_dir, "primary_ridge_permutation.csv"))) perm <- read.csv(file.path(out_dir, "primary_ridge_permutation.csv"))
  optional_table <- function(name) {
    path <- file.path(out_dir, name)
    if (file.exists(path)) read.csv(path, check.names = FALSE) else NULL
  }
  opt_table <- optional_table("primary_ridge_optimism_bootstrap.csv")
  sens_table <- optional_table("ridge_sensitivity_metrics.csv")
  bootstrap_done <- if (is.null(ci)) 0L else min(ci$bootstrap_n)
  optimism_done <- if (is.null(opt_table)) 0L else length(unique(opt_table$bootstrap))
  permutation_done <- if (is.null(perm) || !("kind" %in% names(perm))) 0L else length(unique(perm$permutation[perm$kind == "null"]))
  primary_done <- if (is.null(aggregate_primary)) 0L else min(aggregate_primary$n_repeats)
  row_keys <- function(x, columns) if (is.null(x)) character(0) else unname(apply(x[, columns, drop = FALSE], 1, paste, collapse = "|"))
  expected_primary <- expand.grid(block = names(model_blocks()), model = unlist(cfg$recurrence$models), horizon_days = c(1095, 1825), stringsAsFactors = FALSE)
  primary_columns <- c("block", "model", "horizon_days")
  exact_primary <- !is.null(aggregate_primary) && !anyDuplicated(row_keys(aggregate_primary, primary_columns)) &&
    identical(sort(row_keys(aggregate_primary, primary_columns)), sort(row_keys(expected_primary, primary_columns))) &&
    all(aggregate_primary$n_repeats == reps)
  opt_columns <- c("block", "model", "horizon_days", "bootstrap")
  expected_opt <- expand.grid(block = names(model_blocks()), model = "ridge", horizon_days = c(1095,1825), bootstrap = seq_len(opt_n), stringsAsFactors = FALSE)
  exact_opt <- !is.null(opt_table) && !anyDuplicated(row_keys(opt_table, opt_columns)) &&
    identical(sort(row_keys(opt_table, opt_columns)), sort(row_keys(expected_opt, opt_columns)))
  exact_perm <- !is.null(perm) && "kind" %in% names(perm) && all(perm$kind == "null") &&
    identical(sort(as.integer(perm$permutation)), seq_len(perm_n)) && length(unique(perm$observed)) == 1L &&
    all(is.finite(perm$null_value)) && length(unique(perm$permutation_hash)) == perm_n
  bootstrap_means <- optional_table("primary_bootstrap_repeat_mean_draws.csv")
  expected_boot <- rbind(
    merge(expected_primary, data.frame(metric = c("uno_c", "auc", "brier", "harrell_c"))),
    transform(subset(expected_primary, horizon_days == 1825), metric = "ibs"))
  bootstrap_columns <- c("block", "model", "horizon_days", "metric")
  primary_ci <- if (is.null(ci)) NULL else ci[ci$block %in% names(model_blocks()), , drop = FALSE]
  exact_ci <- !is.null(primary_ci) && !anyDuplicated(row_keys(primary_ci, bootstrap_columns)) &&
    identical(sort(row_keys(primary_ci, bootstrap_columns)), sort(row_keys(expected_boot, bootstrap_columns))) &&
    all(ci$bootstrap_n == boot_n)
  exact_draws <- !is.null(bootstrap_means) &&
    identical(sort(unique(row_keys(bootstrap_means, bootstrap_columns))), sort(row_keys(expected_boot, bootstrap_columns))) &&
    all(table(row_keys(bootstrap_means, bootstrap_columns)) == boot_n) &&
    !anyDuplicated(row_keys(bootstrap_means, c(bootstrap_columns, "draw")))
  all_done <- exact_primary && exact_opt && exact_perm && exact_ci && exact_draws &&
    bootstrap_done == boot_n && optimism_done == opt_n && permutation_done == perm_n &&
    !is.null(sens_table) && length(unique(sens_table$sensitivity)) == 7L && nrow(sens_table) == 14L &&
    all(sens_table$n_repeats == if (quick) 1 else reps)
  if (!identical(expected_fingerprints, source_fingerprints())) stop("Source/input changed during complete run")
  summary <- list(
    status = if (quick) "smoke" else if (all_done) "complete" else "partial",
    completion_checks = list(primary = exact_primary, optimism = exact_opt,
                             permutation = exact_perm, bootstrap_ci = exact_ci,
                             bootstrap_draws = exact_draws, all = all_done),
    source_hashes = as.list(expected_fingerprints),
    outer_repeats = primary_done,
    bootstrap_replicates = bootstrap_done,
    optimism_bootstrap = optimism_done,
    permutations = permutation_done,
    generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
    seed = seed,
    quick = quick,
    recurrence_config = list(
      outer_repeats = reps,
      outer_folds = cfg$recurrence$outer_folds,
      inner_folds = cfg$recurrence$inner_folds,
      bootstrap_B = boot_n,
      optimism_B = opt_n,
      permutation_B = perm_n,
      time_grid_start = min(time_grid),
      time_grid_end = max(time_grid),
      time_grid_n = length(time_grid)
    ),
    input = list(file = "data/ml/recurrence.tsv", sha256 = sha256_file(file.path(ROOT, "data", "ml", "recurrence.tsv")),
                 n_all = nrow(d0), events_all = sum(d0$event), n_primary = nrow(primary), events_primary = sum(primary$event),
                 events_primary_3y = sum(primary$event == 1 & primary$duration_days <= 1095),
                 events_primary_5y = sum(primary$event == 1 & primary$duration_days <= 1825)),
    primary = aggregate_primary,
    bootstrap_ci = ci,
    permutation = if (is.null(perm)) NULL else list(B = perm_n, p_value = unique(perm$p_value), observed = unique(perm$observed)),
    notes = c("Risk scores are out-of-fold only.", "Recorded endpoint is recurrence; no death data were available, so no CIF or OS/DFS/RFS claim is made.")
  )
  write(toJSON(summary, pretty = TRUE, dataframe = "rows", auto_unbox = TRUE, na = "null"),
        file.path(out_dir, "summary.json"))
  write_stage_manifest(paste0("summary_", stage), expected_fingerprints)
  cat(toJSON(summary$input, pretty = TRUE, auto_unbox = TRUE), "\n")
}

if (Sys.getenv("ML_RECURRENCE_SOURCE_ONLY", "0") != "1") main()
