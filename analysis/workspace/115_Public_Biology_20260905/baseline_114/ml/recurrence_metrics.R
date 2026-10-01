## Leakage-controlled recurrence metric utilities.
##
## This file is intentionally standalone so the recurrence model runner can
## source it without bringing in modelling dependencies.

support_floor_default <- 0.1

finite_or_na <- function(x) {
  x[!is.finite(x)] <- NA_real_
  x
}

censor_km <- function(time, event) {
  time <- as.numeric(time)
  event <- as.integer(event)
  stopifnot(length(time) == length(event), all(event %in% c(0L, 1L)))
  ord_times <- sort(unique(time))
  before <- numeric(length(ord_times))
  after <- numeric(length(ord_times))
  surv <- 1
  for (i in seq_along(ord_times)) {
    tt <- ord_times[i]
    before[i] <- surv
    at_risk <- sum(time >= tt)
    censored <- sum(time == tt & event == 0L)
    if (at_risk > 0 && censored > 0) {
      surv <- surv * (1 - censored / at_risk)
    }
    after[i] <- surv
  }
  structure(list(time = ord_times, before = before, after = after), class = "censor_km")
}

km_step <- function(km, t, side = c("after", "before")) {
  side <- match.arg(side)
  t <- as.numeric(t)
  out <- rep(1, length(t))
  if (length(km$time) == 0) return(out)
  if (side == "after") {
    idx <- findInterval(t, km$time)
    ok <- idx > 0
    out[ok] <- km$after[idx[ok]]
    return(out)
  }
  idx <- findInterval(t, km$time)
  exact <- match(t, km$time, nomatch = 0L)
  exact_ok <- exact > 0L
  out[exact_ok] <- km$before[exact[exact_ok]]
  between_ok <- !exact_ok & idx > 0L
  out[between_ok] <- km$after[idx[between_ok]]
  out
}

g_at <- function(km, t) km_step(km, t, "after")

g_before <- function(km, t) km_step(km, t, "before")

breslow_baseline <- function(time, event, linear_predictor, center = NULL) {
  time <- as.numeric(time)
  event <- as.integer(event)
  lp <- as.numeric(linear_predictor)
  stopifnot(length(time) == length(event), length(time) == length(lp))
  if (is.null(center)) center <- mean(lp)
  risk <- exp(lp - center)
  event_times <- sort(unique(time[event == 1L]))
  hazard <- numeric(length(event_times))
  cumulative <- 0
  for (i in seq_along(event_times)) {
    tt <- event_times[i]
    deaths <- sum(time == tt & event == 1L)
    denom <- sum(risk[time >= tt])
    if (!is.finite(denom) || denom <= 0) stop("risk set has non-positive denominator")
    cumulative <- cumulative + deaths / denom
    hazard[i] <- cumulative
  }
  data.frame(time = event_times, hazard = hazard)
}

baseline_hazard_at <- function(baseline, t) {
  t <- as.numeric(t)
  out <- numeric(length(t))
  if (nrow(baseline) == 0) return(out)
  idx <- findInterval(t, baseline$time)
  ok <- idx > 0
  out[ok] <- baseline$hazard[idx[ok]]
  out
}

cox_survival <- function(baseline, linear_predictor, time, center = 0) {
  exp(-baseline_hazard_at(baseline, time) * exp(as.numeric(linear_predictor) - center))
}

cox_risk <- function(baseline, linear_predictor, time, center = 0) {
  1 - cox_survival(baseline, linear_predictor, time, center)
}

row_weights <- function(frame, weights = NULL) {
  if (!is.null(weights)) return(as.numeric(weights))
  if ("weight" %in% names(frame)) return(as.numeric(frame$weight))
  rep(1, nrow(frame))
}

uno_counts_fold <- function(time, event, score, horizon, g_time,
                            weights = NULL, support_floor = support_floor_default) {
  time <- as.numeric(time)
  event <- as.integer(event)
  score <- as.numeric(score)
  g_time <- as.numeric(g_time)
  weights <- if (is.null(weights)) rep(1, length(time)) else as.numeric(weights)
  numerator <- 0
  denominator <- 0
  unsupported <- FALSE
  for (i in seq_along(time)) {
    if (event[i] != 1L || time[i] > horizon) next
    if (weights[i] <= 0) next
    if (!is.finite(g_time[i]) || g_time[i] < support_floor) {
      unsupported <- TRUE
      next
    }
    comp <- which(time > time[i] & weights > 0)
    if (!length(comp)) next
    pair_weight <- (weights[i] * weights[comp]) / (g_time[i] * g_time[i])
    denominator <- denominator + sum(pair_weight)
    numerator <- numerator + sum(pair_weight * ((score[i] > score[comp]) + 0.5 * (score[i] == score[comp])))
  }
  if (unsupported) {
    return(c(estimate = NA_real_, numerator = numerator, denominator = denominator, unsupported = 1))
  }
  c(estimate = if (denominator == 0) NA_real_ else numerator / denominator,
    numerator = numerator, denominator = denominator, unsupported = 0)
}

uno_counts_by_fold <- function(frame, horizon, score_col = "risk_score",
                               g_col = "G_time", weight_col = NULL,
                               support_floor = support_floor_default) {
  numerator <- 0
  denominator <- 0
  unsupported <- FALSE
  for (fold in sort(unique(frame$fold))) {
    z <- frame[frame$fold == fold, , drop = FALSE]
    weights <- if (is.null(weight_col)) row_weights(z) else as.numeric(z[[weight_col]])
    got <- uno_counts_fold(z$duration_days, z$event, z[[score_col]], horizon, z[[g_col]],
                           weights = weights, support_floor = support_floor)
    unsupported <- unsupported || isTRUE(got[["unsupported"]] > 0)
    numerator <- numerator + got[["numerator"]]
    denominator <- denominator + got[["denominator"]]
  }
  c(estimate = if (unsupported || denominator == 0) NA_real_ else numerator / denominator,
    numerator = numerator, denominator = denominator, unsupported = as.integer(unsupported))
}

harrell_counts_fold <- function(time, event, score, weights = NULL) {
  time <- as.numeric(time)
  event <- as.integer(event)
  score <- as.numeric(score)
  weights <- if (is.null(weights)) rep(1, length(time)) else as.numeric(weights)
  ok <- is.finite(time) & is.finite(event) & is.finite(score) & is.finite(weights) & weights > 0
  time <- time[ok]; event <- event[ok]; score <- score[ok]; weights <- weights[ok]
  if (length(time) < 2) return(c(estimate = NA_real_, numerator = 0, denominator = 0))
  numerator <- 0
  denominator <- 0
  for (i in seq_len(length(time) - 1L)) {
    for (j in (i + 1L):length(time)) {
      if (time[i] == time[j]) next
      if (time[i] < time[j] && event[i] == 1L) {
        pw <- weights[i] * weights[j]
        denominator <- denominator + pw
        numerator <- numerator + pw * ((score[i] > score[j]) + 0.5 * (score[i] == score[j]))
      } else if (time[j] < time[i] && event[j] == 1L) {
        pw <- weights[i] * weights[j]
        denominator <- denominator + pw
        numerator <- numerator + pw * ((score[j] > score[i]) + 0.5 * (score[i] == score[j]))
      }
    }
  }
  c(estimate = if (denominator == 0) NA_real_ else numerator / denominator,
    numerator = numerator, denominator = denominator)
}

harrell_counts_by_fold <- function(frame, score_col = "risk_score", weight_col = NULL) {
  numerator <- 0
  denominator <- 0
  for (fold in sort(unique(frame$fold))) {
    z <- frame[frame$fold == fold, , drop = FALSE]
    weights <- if (is.null(weight_col)) row_weights(z) else as.numeric(z[[weight_col]])
    got <- harrell_counts_fold(z$duration_days, z$event, z[[score_col]], weights)
    numerator <- numerator + got[["numerator"]]
    denominator <- denominator + got[["denominator"]]
  }
  c(estimate = if (denominator == 0) NA_real_ else numerator / denominator,
    numerator = numerator, denominator = denominator)
}

time_auc <- function(time, event, score, horizon, g_time, g_horizon,
                     weights = NULL, support_floor = support_floor_default) {
  time <- as.numeric(time)
  event <- as.integer(event)
  score <- as.numeric(score)
  g_time <- as.numeric(g_time)
  g_horizon <- as.numeric(g_horizon)
  weights <- if (is.null(weights)) rep(1, length(time)) else as.numeric(weights)
  cases <- which(event == 1L & time <= horizon & weights > 0)
  controls <- which(time > horizon & weights > 0)
  if (!length(cases) || !length(controls)) return(NA_real_)
  if (any(!is.finite(g_time[cases]) | g_time[cases] < support_floor)) return(NA_real_)
  if (any(!is.finite(g_horizon[controls]) | g_horizon[controls] < support_floor)) return(NA_real_)
  numerator <- 0
  denominator <- 0
  for (i in cases) {
    valid_controls <- controls
    pair_weight <- (weights[i] * weights[valid_controls]) / (g_time[i] * g_horizon[valid_controls])
    denominator <- denominator + sum(pair_weight)
    numerator <- numerator + sum(pair_weight * ((score[i] > score[valid_controls]) + 0.5 * (score[i] == score[valid_controls])))
  }
  if (denominator == 0) NA_real_ else numerator / denominator
}

ipcw_brier <- function(time, event, event_probability, horizon, g_time, g_horizon,
                       weights = NULL, support_floor = support_floor_default) {
  time <- as.numeric(time)
  event <- as.integer(event)
  prob <- pmin(pmax(as.numeric(event_probability), 0), 1)
  g_time <- as.numeric(g_time)
  g_horizon <- as.numeric(g_horizon)
  weights <- if (is.null(weights)) rep(1, length(time)) else as.numeric(weights)
  weighted_error <- 0
  weight_sum <- 0
  event_known <- event == 1L & time <= horizon
  control_known <- time > horizon
  known <- event_known | control_known
  if (any(weights[known] > 0 & event_known[known] & (!is.finite(g_time[known]) | g_time[known] < support_floor))) return(NA_real_)
  if (any(weights[known] > 0 & control_known[known] & (!is.finite(g_horizon[known]) | g_horizon[known] < support_floor))) return(NA_real_)
  for (i in which(event_known | control_known)) {
    if (weights[i] <= 0) next
    if (event_known[i]) {
      g <- g_time[i]
      target <- 1
    } else {
      g <- g_horizon[i]
      target <- 0
    }
    if (!is.finite(g) || g < support_floor) next
    w <- weights[i] / g
    weighted_error <- weighted_error + w * (target - prob[i])^2
  }
  weight_sum <- sum(weights)
  if (weight_sum == 0) NA_real_ else weighted_error / weight_sum
}

calibration_horizon <- function(time, event, event_probability, horizon,
                                g_time, g_horizon, weights = NULL,
                                support_floor = support_floor_default,
                                epsilon = 1e-6) {
  time <- as.numeric(time)
  event <- as.integer(event)
  prob <- pmin(pmax(as.numeric(event_probability), epsilon), 1 - epsilon)
  g_time <- as.numeric(g_time)
  g_horizon <- as.numeric(g_horizon)
  weights <- if (is.null(weights)) rep(1, length(time)) else as.numeric(weights)
  event_known <- event == 1L & time <= horizon
  control_known <- time > horizon
  known <- event_known | control_known
  if (any(weights[known] > 0 & event_known[known] & (!is.finite(g_time[known]) | g_time[known] < support_floor))) {
    return(data.frame(intercept = NA_real_, slope = NA_real_, n_used = sum(known),
                      n_events = sum(event_known), weight_sum = NA_real_,
                      status = "nonestimable_unsupported_g", stringsAsFactors = FALSE))
  }
  if (any(weights[known] > 0 & control_known[known] & (!is.finite(g_horizon[known]) | g_horizon[known] < support_floor))) {
    return(data.frame(intercept = NA_real_, slope = NA_real_, n_used = sum(known),
                      n_events = sum(event_known), weight_sum = NA_real_,
                      status = "nonestimable_unsupported_g", stringsAsFactors = FALSE))
  }
  y <- as.integer(event_known[known])
  p <- prob[known]
  base_weights <- weights[known]
  ipcw <- ifelse(y == 1L, 1 / g_time[known], 1 / g_horizon[known])
  fit_weights <- base_weights * ipcw
  keep <- is.finite(y) & is.finite(p) & is.finite(fit_weights) & fit_weights > 0
  y <- y[keep]
  p <- p[keep]
  fit_weights <- fit_weights[keep]
  lp <- qlogis(p)
  status_frame <- function(status) {
    data.frame(intercept = NA_real_, slope = NA_real_, n_used = length(y),
               n_events = sum(y), weight_sum = sum(fit_weights),
               status = status, stringsAsFactors = FALSE)
  }
  if (length(y) < 2L || length(unique(y)) < 2L) return(status_frame("nonestimable_one_class"))
  if (!is.finite(stats::sd(lp)) || stats::sd(lp) == 0) return(status_frame("nonestimable_constant_prediction"))
  lp_case <- lp[y == 1L]
  lp_control <- lp[y == 0L]
  if (min(lp_case) >= max(lp_control) || min(lp_control) >= max(lp_case)) {
    return(status_frame("nonestimable_separation"))
  }
  warn <- NULL
  fit <- withCallingHandlers(
    tryCatch(stats::glm(y ~ lp, family = stats::binomial(), weights = fit_weights,
                        control = stats::glm.control(maxit = 50)),
             error = function(e) e),
    warning = function(w) {
      warn <<- conditionMessage(w)
      invokeRestart("muffleWarning")
    }
  )
  if (inherits(fit, "error") || !isTRUE(fit$converged)) return(status_frame("nonestimable_nonconvergence"))
  co <- stats::coef(fit)
  if (length(co) != 2L || any(!is.finite(co))) return(status_frame("nonestimable_nonfinite"))
  if (!is.null(warn) && grepl("fitted probabilities numerically 0 or 1|algorithm did not converge", warn)) {
    return(status_frame("nonestimable_nonconvergence"))
  }
  data.frame(intercept = unname(co[[1]]), slope = unname(co[[2]]),
             n_used = length(y), n_events = sum(y), weight_sum = sum(fit_weights),
             status = "estimable", stringsAsFactors = FALSE)
}

ibs <- function(frame, max_horizon = 1825, n_grid = 101, risk_prefix = "risk_grid_",
                support_floor = support_floor_default, weight_col = NULL) {
  grid <- seq(0, max_horizon, length.out = n_grid)
  vals <- rep(NA_real_, length(grid))
  weights <- if (is.null(weight_col)) row_weights(frame) else as.numeric(frame[[weight_col]])
  for (i in seq_along(grid)) {
    tau <- grid[i]
    risk_col <- paste0(risk_prefix, i)
    g_col <- paste0("G_grid_", i)
    if (!(risk_col %in% names(frame)) || !(g_col %in% names(frame))) next
    vals[i] <- ipcw_brier(frame$duration_days, frame$event, frame[[risk_col]], tau,
                          frame$G_time, frame[[g_col]], weights, support_floor)
  }
  if (any(is.na(vals))) return(NA_real_)
  sum(diff(grid) * (head(vals, -1L) + tail(vals, -1L)) / 2) / max_horizon
}

make_bootstrap_weights <- function(patient_ids, B = 2000, seed = 20260905) {
  patient_ids <- sort(unique(as.character(patient_ids)))
  set.seed(seed)
  W <- matrix(0, nrow = B, ncol = length(patient_ids))
  colnames(W) <- patient_ids
  for (b in seq_len(B)) {
    counts <- table(sample(patient_ids, length(patient_ids), replace = TRUE))
    W[b, names(counts)] <- as.numeric(counts)
  }
  W
}

matrix_ratio <- function(W, numerator_kernel, denominator_kernel) {
  numerator <- rowSums((W %*% numerator_kernel) * W)
  denominator <- rowSums((W %*% denominator_kernel) * W)
  ifelse(denominator == 0, NA_real_, numerator / denominator)
}

precompute_fold_pair_kernel <- function(frame, horizon, score_col = "risk_score",
                                        metric = c("uno_c", "auc", "harrell_c"),
                                        support_floor = support_floor_default) {
  metric <- match.arg(metric)
  patients <- sort(unique(as.character(frame$study_id)))
  p <- length(patients)
  numerator <- matrix(0, p, p, dimnames = list(patients, patients))
  denominator <- matrix(0, p, p, dimnames = list(patients, patients))
  unsupported <- FALSE

  fold_values <- if (metric == "auc") NA else sort(unique(frame$fold))
  for (fold in fold_values) {
    if (is.na(fold)) {
      z <- frame
    } else {
      z <- frame[frame$fold == fold, , drop = FALSE]
    }
    ids <- as.character(z$study_id)
    idx <- match(ids, patients)
    time <- as.numeric(z$duration_days)
    event <- as.integer(z$event)
    score <- as.numeric(z[[score_col]])
    if (metric == "uno_c") {
      g_time <- as.numeric(z$G_time)
      for (i in seq_along(time)) {
        if (event[i] != 1L || time[i] > horizon) next
        if (!is.finite(g_time[i]) || g_time[i] < support_floor) {
          unsupported <- TRUE
          next
        }
        comp <- which(time > time[i])
        if (!length(comp)) next
        w <- 1 / (g_time[i] * g_time[i])
        denominator[idx[i], idx[comp]] <- denominator[idx[i], idx[comp]] + w
        numerator[idx[i], idx[comp]] <- numerator[idx[i], idx[comp]] +
          w * ((score[i] > score[comp]) + 0.5 * (score[i] == score[comp]))
      }
    } else if (metric == "auc") {
      g_col <- paste0("G_", as.integer(horizon))
      cases <- which(event == 1L & time <= horizon)
      controls <- which(time > horizon)
      if (!length(cases) || !length(controls)) next
      if (any(!is.finite(z$G_time[cases]) | z$G_time[cases] < support_floor) ||
          any(!is.finite(z[[g_col]][controls]) | z[[g_col]][controls] < support_floor)) {
        unsupported <- TRUE
        next
      }
      for (i in cases) {
        w <- 1 / (z$G_time[i] * z[[g_col]][controls])
        denominator[idx[i], idx[controls]] <- denominator[idx[i], idx[controls]] + w
        numerator[idx[i], idx[controls]] <- numerator[idx[i], idx[controls]] +
          w * ((score[i] > score[controls]) + 0.5 * (score[i] == score[controls]))
      }
    } else {
      n <- length(time)
      if (n < 2L) next
      for (i in seq_len(n - 1L)) {
        for (j in (i + 1L):n) {
          if (time[i] == time[j]) next
          if (time[i] < time[j] && event[i] == 1L) {
            denominator[idx[i], idx[j]] <- denominator[idx[i], idx[j]] + 1
            numerator[idx[i], idx[j]] <- numerator[idx[i], idx[j]] +
              ((score[i] > score[j]) + 0.5 * (score[i] == score[j]))
          } else if (time[j] < time[i] && event[j] == 1L) {
            denominator[idx[j], idx[i]] <- denominator[idx[j], idx[i]] + 1
            numerator[idx[j], idx[i]] <- numerator[idx[j], idx[i]] +
              ((score[j] > score[i]) + 0.5 * (score[i] == score[j]))
          }
        }
      }
    }
  }
  list(metric = metric, horizon = horizon, patients = patients,
       numerator = numerator, denominator = denominator, unsupported = unsupported)
}

precompute_brier_vector <- function(frame, horizon, risk_col = NULL,
                                    g_horizon_col = NULL,
                                    support_floor = support_floor_default) {
  if (is.null(risk_col)) risk_col <- paste0("risk_", as.integer(horizon))
  if (is.null(g_horizon_col)) g_horizon_col <- paste0("G_", as.integer(horizon))
  patients <- sort(unique(as.character(frame$study_id)))
  errors <- setNames(numeric(length(patients)), patients)
  unsupported <- FALSE
  time <- as.numeric(frame$duration_days)
  event <- as.integer(frame$event)
  prob <- pmin(pmax(as.numeric(frame[[risk_col]]), 0), 1)
  known_event <- event == 1L & time <= horizon
  known_control <- time > horizon
  known <- known_event | known_control
  bad_event <- known_event & (!is.finite(frame$G_time) | frame$G_time < support_floor)
  bad_control <- known_control & (!is.finite(frame[[g_horizon_col]]) | frame[[g_horizon_col]] < support_floor)
  if (any(bad_event | bad_control)) unsupported <- TRUE
  for (i in which(known & !(bad_event | bad_control))) {
    id <- as.character(frame$study_id[i])
    if (known_event[i]) {
      errors[id] <- errors[id] + ((1 - prob[i])^2) / frame$G_time[i]
    } else {
      errors[id] <- errors[id] + (prob[i]^2) / frame[[g_horizon_col]][i]
    }
  }
  list(metric = "brier", horizon = horizon, patients = patients,
       errors = errors, unsupported = unsupported)
}

precompute_ibs_vector <- function(frame, max_horizon = 1825, n_grid = 101,
                                  risk_prefix = "risk_grid_",
                                  support_floor = support_floor_default) {
  patients <- sort(unique(as.character(frame$study_id)))
  grid <- seq(0, max_horizon, length.out = n_grid)
  by_grid <- matrix(NA_real_, nrow = length(patients), ncol = length(grid),
                    dimnames = list(patients, paste0("t", seq_along(grid))))
  unsupported <- FALSE
  for (i in seq_along(grid)) {
    risk_col <- paste0(risk_prefix, i)
    g_col <- paste0("G_grid_", i)
    if (!(risk_col %in% names(frame)) || !(g_col %in% names(frame))) {
      unsupported <- TRUE
      next
    }
    vec <- precompute_brier_vector(frame, grid[i], risk_col = risk_col,
                                   g_horizon_col = g_col,
                                   support_floor = support_floor)
    by_grid[vec$patients, i] <- vec$errors
    unsupported <- unsupported || vec$unsupported
  }
  if (unsupported || anyNA(by_grid)) {
    errors <- setNames(rep(NA_real_, length(patients)), patients)
  } else {
    errors <- setNames(numeric(length(patients)), patients)
    for (i in seq_len(length(grid) - 1L)) {
      errors <- errors + (grid[i + 1L] - grid[i]) * (by_grid[, i] + by_grid[, i + 1L]) / 2
    }
    errors <- errors / max_horizon
  }
  list(metric = "ibs", horizon = max_horizon, patients = patients,
       errors = errors, unsupported = unsupported)
}

bootstrap_values_from_kernel <- function(kernel, W) {
  common <- intersect(colnames(W), kernel$patients)
  WW <- W[, common, drop = FALSE]
  if (isTRUE(kernel$unsupported)) return(rep(NA_real_, nrow(WW)))
  matrix_ratio(WW, kernel$numerator[common, common, drop = FALSE],
               kernel$denominator[common, common, drop = FALSE])
}

bootstrap_values_from_vector <- function(vector_obj, W, denominator = NULL) {
  common <- intersect(colnames(W), vector_obj$patients)
  WW <- W[, common, drop = FALSE]
  if (isTRUE(vector_obj$unsupported) || anyNA(vector_obj$errors[common])) {
    return(rep(NA_real_, nrow(WW)))
  }
  den <- if (is.null(denominator)) rowSums(WW) else denominator
  ifelse(den == 0, NA_real_, as.numeric(WW %*% vector_obj$errors[common]) / den)
}

observed_metric_points <- function(oof, metrics = c("uno_c", "auc", "brier", "harrell_c"),
                                   horizons = c(1095, 1825),
                                   support_floor = support_floor_default) {
  base_metrics <- intersect(metrics, c("uno_c", "auc", "brier", "harrell_c"))
  rows <- list()
  ix <- 1L
  if (length(base_metrics)) {
    repmet <- repeat_metrics_from_oof(oof, horizons = horizons, support_floor = support_floor)
    for (block in sort(unique(repmet$block))) {
      for (model in sort(unique(repmet$model))) {
        for (horizon in sort(unique(repmet$horizon_days))) {
          z <- repmet[repmet$block == block & repmet$model == model &
                        repmet$horizon_days == horizon, , drop = FALSE]
          for (metric in base_metrics) {
            values <- z[[metric]]
            rows[[ix]] <- data.frame(
              block = block, model = model, horizon_days = horizon, metric = metric,
              observed = if (all(is.finite(values))) mean(values) else NA_real_,
              observed_n_estimable = sum(is.finite(values)),
              observed_status = if (all(is.finite(values))) "estimable" else "nonestimable",
              stringsAsFactors = FALSE
            )
            ix <- ix + 1L
          }
        }
      }
    }
  }
  if ("ibs" %in% metrics) {
    keys <- unique(oof[, c("repeat", "block", "model"), drop = FALSE])
    ibs_rows <- list()
    bx <- 1L
    for (i in seq_len(nrow(keys))) {
      z <- oof[oof[["repeat"]] == keys[["repeat"]][i] &
                 oof$block == keys$block[i] &
                 oof$model == keys$model[i], , drop = FALSE]
      ibs_rows[[bx]] <- data.frame(
        block = keys$block[i],
        model = keys$model[i],
        horizon_days = 1825,
        metric = "ibs",
        value = ibs(z, max_horizon = 1825, n_grid = 101, support_floor = support_floor),
        stringsAsFactors = FALSE
      )
      bx <- bx + 1L
    }
    ibs_frame <- do.call(rbind, ibs_rows)
    for (block in sort(unique(ibs_frame$block))) {
      for (model in sort(unique(ibs_frame$model))) {
        z <- ibs_frame[ibs_frame$block == block & ibs_frame$model == model, , drop = FALSE]
        rows[[ix]] <- data.frame(
          block = block, model = model, horizon_days = 1825, metric = "ibs",
          observed = if (all(is.finite(z$value))) mean(z$value) else NA_real_,
          observed_n_estimable = sum(is.finite(z$value)),
          observed_status = if (all(is.finite(z$value))) "estimable" else "nonestimable",
          stringsAsFactors = FALSE
        )
        ix <- ix + 1L
      }
    }
  }
  if (!length(rows)) {
    return(data.frame(block = character(), model = character(), horizon_days = integer(),
                      metric = character(), observed = numeric(),
                      observed_n_estimable = integer(), observed_status = character()))
  }
  do.call(rbind, rows)
}

fast_bootstrap_ci <- function(oof, metrics = c("uno_c", "auc", "brier", "harrell_c"),
                              B = 2000, seed = 20260905, horizons = c(1095, 1825),
                              support_floor = support_floor_default,
                              include_deltas = TRUE) {
  W <- make_bootstrap_weights(oof$study_id, B = B, seed = seed)
  keys <- unique(oof[, c("repeat", "block", "model"), drop = FALSE])
  draws <- list()
  ix <- 1L
  for (i in seq_len(nrow(keys))) {
    z <- oof[oof[["repeat"]] == keys[["repeat"]][i] &
               oof$block == keys$block[i] &
               oof$model == keys$model[i], , drop = FALSE]
    for (horizon in horizons) {
      for (metric in metrics) {
        values <- rep(NA_real_, B)
        if (metric %in% c("uno_c", "auc", "harrell_c")) {
          kernel <- precompute_fold_pair_kernel(z, horizon, metric = metric,
                                                score_col = if (metric == "auc") paste0("risk_", as.integer(horizon)) else "risk_score",
                                                support_floor = support_floor)
          values <- bootstrap_values_from_kernel(kernel, W)
        } else if (metric == "brier") {
          vec <- precompute_brier_vector(z, horizon, support_floor = support_floor)
          values <- bootstrap_values_from_vector(vec, W)
        } else if (metric == "ibs") {
          if (horizon != 1825) next
          vec <- precompute_ibs_vector(z, max_horizon = 1825, n_grid = 101,
                                       support_floor = support_floor)
          values <- bootstrap_values_from_vector(vec, W)
        }
        draws[[ix]] <- data.frame(
          draw = seq_len(B),
          repeat_id = keys[["repeat"]][i],
          block = keys$block[i],
          model = keys$model[i],
          horizon_days = horizon,
          metric = metric,
          value = values,
          stringsAsFactors = FALSE
        )
        ix <- ix + 1L
      }
    }
  }
  draw_frame <- do.call(rbind, draws)
  names(draw_frame)[names(draw_frame) == "repeat_id"] <- "repeat"
  mean_draws <- aggregate(value ~ draw + block + model + horizon_days + metric,
                          data = draw_frame, FUN = function(x) if (all(is.finite(x))) mean(x) else NA_real_,
                          na.action = na.pass)
  observed <- observed_metric_points(oof, metrics = metrics, horizons = horizons,
                                     support_floor = support_floor)
  rows <- by(mean_draws, mean_draws[c("block", "model", "horizon_days", "metric")], function(z) {
    ok <- is.finite(z$value)
    obs <- observed[observed$block == z$block[1] & observed$model == z$model[1] &
                      observed$horizon_days == z$horizon_days[1] & observed$metric == z$metric[1], ]
    data.frame(
      block = z$block[1],
      model = z$model[1],
      horizon_days = z$horizon_days[1],
      metric = z$metric[1],
      estimate = if (nrow(obs)) obs$observed[1] else NA_real_,
      ci_low = if (all(ok)) as.numeric(quantile(z$value, 0.025, names = FALSE)) else NA_real_,
      ci_high = if (all(ok)) as.numeric(quantile(z$value, 0.975, names = FALSE)) else NA_real_,
      bootstrap_n = B,
      n_estimable = sum(ok),
      status = if (all(ok)) "estimable" else "nonestimable",
      stringsAsFactors = FALSE
    )
  })
  ci <- do.call(rbind, rows)
  delta_ci <- NULL
  if (include_deltas && all(c("combined", "clinical") %in% unique(mean_draws$block))) {
    delta_rows <- list()
    dx <- 1L
    for (horizon in horizons) {
      for (metric in intersect(c("uno_c", "brier"), metrics)) {
        a <- mean_draws[mean_draws$block == "combined" & mean_draws$model == "ridge" &
                          mean_draws$horizon_days == horizon & mean_draws$metric == metric, ]
        b <- mean_draws[mean_draws$block == "clinical" & mean_draws$model == "ridge" &
                          mean_draws$horizon_days == horizon & mean_draws$metric == metric, ]
        if (!nrow(a) || !nrow(b)) next
        m <- merge(a[, c("draw", "value")], b[, c("draw", "value")],
                   by = "draw", suffixes = c("_combined", "_clinical"))
        delta <- m$value_combined - m$value_clinical
        ok <- is.finite(delta)
        oa <- observed[observed$block == "combined" & observed$model == "ridge" &
                         observed$horizon_days == horizon & observed$metric == metric, ]
        ob <- observed[observed$block == "clinical" & observed$model == "ridge" &
                         observed$horizon_days == horizon & observed$metric == metric, ]
        observed_delta <- if (nrow(oa) && nrow(ob) && is.finite(oa$observed[1]) && is.finite(ob$observed[1])) {
          oa$observed[1] - ob$observed[1]
        } else {
          NA_real_
        }
        delta_rows[[dx]] <- data.frame(
          block = "combined_minus_clinical",
          model = "ridge",
          horizon_days = horizon,
          metric = paste0("delta_", metric),
          estimate = observed_delta,
          ci_low = if (all(ok)) as.numeric(quantile(delta, 0.025, names = FALSE)) else NA_real_,
          ci_high = if (all(ok)) as.numeric(quantile(delta, 0.975, names = FALSE)) else NA_real_,
          bootstrap_n = B,
          n_estimable = sum(ok),
          status = if (all(ok)) "estimable" else "nonestimable",
          stringsAsFactors = FALSE
        )
        dx <- dx + 1L
      }
    }
    if (length(delta_rows)) delta_ci <- do.call(rbind, delta_rows)
  }
  list(ci = if (is.null(delta_ci)) ci else rbind(ci, delta_ci),
       draws = draw_frame,
       repeat_mean_draws = mean_draws,
       W = W)
}

repeat_metrics_from_oof <- function(oof, horizons = c(1095, 1825),
                                    score_col = "risk_score",
                                    support_floor = support_floor_default,
                                    weight_col = NULL) {
  required <- c("repeat", "fold", "block", "model", "duration_days", "event", score_col, "G_time")
  missing <- setdiff(required, names(oof))
  if (length(missing)) stop("OOF frame missing columns: ", paste(missing, collapse = ", "))
  rows <- list()
  ix <- 1L
  for (r in sort(unique(oof[["repeat"]]))) {
    for (block in sort(unique(oof$block))) {
      for (model in sort(unique(oof$model))) {
        z <- oof[oof[["repeat"]] == r & oof$block == block & oof$model == model, , drop = FALSE]
        weights <- if (is.null(weight_col)) row_weights(z) else as.numeric(z[[weight_col]])
        for (horizon in horizons) {
          risk_col <- paste0("risk_", as.integer(horizon))
          g_horizon_col <- paste0("G_", as.integer(horizon))
          if (!(risk_col %in% names(z)) || !(g_horizon_col %in% names(z))) {
            stop("OOF frame missing horizon columns for ", horizon)
          }
          uno <- uno_counts_by_fold(z, horizon, score_col, "G_time", weight_col, support_floor)
          hc <- harrell_counts_by_fold(z, score_col, weight_col)
          auc <- time_auc(z$duration_days, z$event, z[[risk_col]], horizon,
                          z$G_time, z[[g_horizon_col]], weights, support_floor)
          brier <- ipcw_brier(z$duration_days, z$event, z[[risk_col]], horizon,
                              z$G_time, z[[g_horizon_col]], weights, support_floor)
          cal <- calibration_horizon(z$duration_days, z$event, z[[risk_col]], horizon,
                                     z$G_time, z[[g_horizon_col]], weights, support_floor)
          rows[[ix]] <- data.frame(
            repeat_id = r,
            block = block,
            model = model,
            horizon_days = horizon,
            uno_c = uno[["estimate"]],
            uno_num = uno[["numerator"]],
            uno_den = uno[["denominator"]],
            uno_unsupported = uno[["unsupported"]],
            harrell_c = hc[["estimate"]],
            harrell_num = hc[["numerator"]],
            harrell_den = hc[["denominator"]],
            auc = auc,
            brier = brier,
            cal_intercept = cal$intercept,
            cal_slope = cal$slope,
            cal_status = cal$status,
            cal_n_used = cal$n_used,
            cal_n_events = cal$n_events,
            cal_weight_sum = cal$weight_sum,
            estimable = is.finite(uno[["estimate"]]) && is.finite(auc) && is.finite(brier),
            stringsAsFactors = FALSE
          )
          ix <- ix + 1L
        }
      }
    }
  }
  out <- do.call(rbind, rows)
  names(out)[names(out) == "repeat_id"] <- "repeat"
  out
}

aggregate_repeat_metrics <- function(repeat_metrics) {
  keys <- c("block", "model", "horizon_days")
  out <- by(repeat_metrics, repeat_metrics[keys], function(z) {
    metric_mean <- function(v) if (all(is.finite(v))) mean(v) else NA_real_
    data.frame(
      block = z$block[1],
      model = z$model[1],
      horizon_days = z$horizon_days[1],
      n_repeats = nrow(z),
      estimable_repeats = sum(z$estimable %in% TRUE),
      uno_c_mean = metric_mean(z$uno_c),
      uno_c_sd = if (all(is.finite(z$uno_c))) sd(z$uno_c) else NA_real_,
      harrell_c_mean = metric_mean(z$harrell_c),
      auc_mean = metric_mean(z$auc),
      brier_mean = metric_mean(z$brier),
      cal_estimable_repeats = sum(z$cal_status == "estimable"),
      cal_intercept_mean = metric_mean(z$cal_intercept),
      cal_slope_mean = metric_mean(z$cal_slope),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, out)
}

add_patient_weights <- function(oof, sampled_ids, group_col = "study_id") {
  counts <- table(sampled_ids)
  out <- oof[oof[[group_col]] %in% names(counts), , drop = FALSE]
  out$weight <- as.numeric(counts[as.character(out[[group_col]])])
  out
}

bootstrap_metric_ci <- function(oof, metric = "uno_c", B = 2000, seed = 20260905,
                                horizons = c(1095, 1825), support_floor = support_floor_default) {
  if (!metric %in% c("uno_c", "auc", "brier", "harrell_c")) stop("unsupported metric")
  set.seed(seed)
  patients <- unique(oof$study_id)
  observed <- repeat_metrics_from_oof(oof, horizons = horizons, support_floor = support_floor)
  keys <- unique(observed[, c("block", "model", "horizon_days"), drop = FALSE])
  boot_values <- matrix(NA_real_, nrow = B, ncol = nrow(keys))
  for (b in seq_len(B)) {
    sampled <- sample(patients, length(patients), replace = TRUE)
    weighted <- add_patient_weights(oof, sampled)
    bm <- repeat_metrics_from_oof(weighted, horizons = horizons,
                                  support_floor = support_floor, weight_col = "weight")
    for (i in seq_len(nrow(keys))) {
      key <- keys[i, , drop = FALSE]
      bz <- merge(key, bm, by = c("block", "model", "horizon_days"))
      boot_values[b, i] <- if (all(is.finite(bz[[metric]]))) mean(bz[[metric]]) else NA_real_
    }
  }
  rows <- vector("list", nrow(keys))
  for (i in seq_len(nrow(keys))) {
    key <- keys[i, , drop = FALSE]
    obs_z <- merge(key, observed, by = c("block", "model", "horizon_days"))
    boot <- boot_values[, i]
    estimate <- if (all(is.finite(obs_z[[metric]]))) mean(obs_z[[metric]]) else NA_real_
    if (any(is.na(boot))) {
      ci_low <- NA_real_
      ci_high <- NA_real_
    } else {
      ci_low <- as.numeric(quantile(boot, 0.025, names = FALSE))
      ci_high <- as.numeric(quantile(boot, 0.975, names = FALSE))
    }
    rows[[i]] <- data.frame(
      block = key$block,
      model = key$model,
      horizon_days = key$horizon_days,
      metric = metric,
      estimate = estimate,
      ci_low = ci_low,
      ci_high = ci_high,
      bootstrap_n = B,
      stringsAsFactors = FALSE
    )
  }
  do.call(rbind, rows)
}
