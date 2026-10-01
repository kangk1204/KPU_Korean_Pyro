library(testthat)

cmd_args <- commandArgs(FALSE)
file_arg <- cmd_args[grepl("^--file=", cmd_args)]
script_dir <- if (length(file_arg)) dirname(normalizePath(sub("^--file=", "", file_arg[1]), mustWork = FALSE)) else "."
metric_candidates <- c(
  file.path("ml", "recurrence_metrics.R"),
  file.path("..", "ml", "recurrence_metrics.R"),
  file.path(script_dir, "..", "ml", "recurrence_metrics.R")
)
metric_path <- metric_candidates[file.exists(metric_candidates)][1]
if (is.na(metric_path)) stop("cannot locate ml/recurrence_metrics.R")
source(metric_path)

test_that("censoring KM exposes left and right step values", {
  km <- censor_km(c(1, 2, 2, 4), c(1, 0, 1, 0))

  expect_equal(g_before(km, 2), 1)
  expect_equal(g_at(km, 2), 2 / 3, tolerance = 1e-12)
  expect_equal(g_before(km, 3), 2 / 3, tolerance = 1e-12)
  expect_equal(g_before(km, 0.5), 1)
})

test_that("Breslow baseline handles tied events with train centering", {
  times <- c(2, 2, 3, 4)
  events <- c(1, 1, 0, 1)
  lp <- log(c(1, 2, 3, 4))
  center <- mean(lp)
  baseline <- breslow_baseline(times, events, lp, center = center)
  risk <- exp(lp - center)

  expect_equal(baseline$time, c(2, 4))
  expect_equal(baseline$hazard[1], 2 / sum(risk), tolerance = 1e-12)
  expect_equal(baseline$hazard[2], 2 / sum(risk) + 1 / risk[4], tolerance = 1e-12)
})

test_that("Cox survival uses step hazard and explicit center", {
  baseline <- breslow_baseline(c(2, 4), c(1, 1), c(0, log(2)), center = 0)
  survival <- cox_survival(baseline, linear_predictor = log(2), time = 3, center = 0)

  expect_equal(survival, exp(-(1 / 3) * 2), tolerance = 1e-12)
})

test_that("Uno C counts tied risk as half concordant", {
  got <- uno_counts_fold(
    time = c(1, 2, 3),
    event = c(1, 1, 0),
    score = c(0.5, 0.5, 0.1),
    horizon = 3,
    g_time = c(1, 1, 1)
  )

  expect_equal(got[["denominator"]], 3)
  expect_equal(got[["numerator"]], 2.5)
  expect_equal(got[["estimate"]], 2.5 / 3, tolerance = 1e-12)
})

test_that("time AUC returns half for constant risks", {
  auc <- time_auc(
    time = c(1, 2, 5, 6),
    event = c(1, 1, 0, 0),
    score = c(0.4, 0.4, 0.4, 0.4),
    horizon = 3,
    g_time = rep(1, 4),
    g_horizon = rep(1, 4)
  )

  expect_equal(auc, 0.5, tolerance = 1e-12)
})

test_that("IPCW Brier excludes early censored observations", {
  brier <- ipcw_brier(
    time = c(1, 2, 5),
    event = c(0, 1, 0),
    event_probability = c(0.99, 0.25, 0.25),
    horizon = 3,
    g_time = c(1, 1, 1),
    g_horizon = c(1, 1, 1)
  )

  expect_equal(brier, (((1 - 0.25)^2) + 0.25^2) / 3, tolerance = 1e-12)
})

test_that("repeat metrics sum Uno counts by held-out fold and pool horizon metrics", {
  oof <- data.frame(
    repeat_id = c(1, 1, 1, 1),
    fold = c(1, 1, 2, 2),
    study_id = paste0("P", 1:4),
    block = "combined",
    model = "ridge",
    duration_days = c(1, 5, 2, 6),
    event = c(1, 0, 1, 0),
    risk_score = c(0.8, 0.2, 0.4, 0.1),
    risk_3 = c(0.8, 0.2, 0.4, 0.1),
    G_time = rep(1, 4),
    G_3 = rep(1, 4)
  )
  names(oof)[names(oof) == "repeat_id"] <- "repeat"
  metrics <- repeat_metrics_from_oof(oof, horizons = 3)

  expect_equal(nrow(metrics), 1)
  expect_equal(metrics$uno_num, 2)
  expect_equal(metrics$uno_den, 2)
  expect_equal(metrics$uno_c, 1)
  expect_equal(metrics$auc, 1)
  expect_equal(metrics$brier, mean(c((1 - 0.8)^2, 0.2^2, (1 - 0.4)^2, 0.1^2)), tolerance = 1e-12)
  expect_true(metrics$estimable)
})

test_that("patient multiplicities affect pair counts without duplicating rows", {
  got <- uno_counts_fold(
    time = c(1, 2, 4),
    event = c(1, 1, 0),
    score = c(0.9, 0.1, 0.5),
    horizon = 5,
    g_time = c(1, 1, 1),
    weights = c(2, 1, 3)
  )

  expect_equal(got[["denominator"]], 11)
  expect_equal(got[["numerator"]], 8)
  expect_equal(got[["estimate"]], 8 / 11, tolerance = 1e-12)
})

test_that("unsupported censoring weights produce non-estimable metrics", {
  got <- uno_counts_fold(
    time = c(1, 2, 3),
    event = c(1, 1, 0),
    score = c(0.9, 0.8, 0.1),
    horizon = 3,
    g_time = c(0.09, 0.09, 1),
    support_floor = 0.1
  )

  expect_true(is.na(got[["estimate"]]))
  expect_equal(got[["denominator"]], 0)
})

test_that("unsupported horizon weights make AUC and Brier non-estimable", {
  auc <- time_auc(
    time = c(1, 5),
    event = c(1, 0),
    score = c(0.9, 0.1),
    horizon = 3,
    g_time = c(1, 1),
    g_horizon = c(1, 0.09)
  )
  brier <- ipcw_brier(
    time = c(1, 5),
    event = c(1, 0),
    event_probability = c(0.9, 0.1),
    horizon = 3,
    g_time = c(1, 1),
    g_horizon = c(1, 0.09)
  )

  expect_true(is.na(auc))
  expect_true(is.na(brier))
})

test_that("calibration uses IPCW weights for known horizon outcomes", {
  time <- c(1, 2, 5, 6)
  event <- c(1, 1, 0, 0)
  prob <- c(0.55, 0.45, 0.40, 0.60)
  g_time <- c(0.5, 0.8, 1, 1)
  g_horizon <- c(1, 1, 0.25, 0.5)
  mult <- c(1, 2, 1, 1)
  got <- calibration_horizon(time, event, prob, 3, g_time, g_horizon, weights = mult)
  y <- c(1, 1, 0, 0)
  lp <- qlogis(prob)
  expected_weights <- mult * c(1 / 0.5, 1 / 0.8, 1 / 0.25, 1 / 0.5)
  expected <- suppressWarnings(glm(y ~ lp, family = binomial(), weights = expected_weights))

  expect_equal(got$status, "estimable")
  expect_equal(got$intercept, unname(coef(expected)[1]), tolerance = 1e-10)
  expect_equal(got$slope, unname(coef(expected)[2]), tolerance = 1e-10)
})

test_that("calibration is non-estimable for separated or constant predictions", {
  separated <- calibration_horizon(
    time = c(1, 2, 5, 6),
    event = c(1, 1, 0, 0),
    event_probability = c(0.8, 0.7, 0.2, 0.1),
    horizon = 3,
    g_time = rep(1, 4),
    g_horizon = rep(1, 4)
  )
  constant <- calibration_horizon(
    time = c(1, 2, 5, 6),
    event = c(1, 1, 0, 0),
    event_probability = rep(0.5, 4),
    horizon = 3,
    g_time = rep(1, 4),
    g_horizon = rep(1, 4)
  )

  expect_equal(separated$status, "nonestimable_separation")
  expect_true(is.na(separated$slope))
  expect_equal(constant$status, "nonestimable_constant_prediction")
  expect_true(is.na(constant$slope))
})

toy_oof <- function() {
  out <- data.frame(
    repeat_id = rep(c(1, 2), each = 8),
    fold = rep(c(1, 1, 2, 2), times = 4),
    study_id = rep(paste0("P", 1:4), times = 4),
    block = rep(c("clinical", "combined"), each = 4, times = 2),
    model = "ridge",
    duration_days = rep(c(1, 5, 2, 6), times = 4),
    event = rep(c(1, 0, 1, 0), times = 4),
    risk_score = c(0.6, 0.3, 0.4, 0.2, 0.8, 0.2, 0.4, 0.1,
                   0.55, 0.35, 0.45, 0.15, 0.75, 0.25, 0.35, 0.05),
    risk_3 = c(0.6, 0.3, 0.4, 0.2, 0.8, 0.2, 0.4, 0.1,
               0.55, 0.35, 0.45, 0.15, 0.75, 0.25, 0.35, 0.05),
    G_time = 1,
    G_3 = 1
  )
  names(out)[names(out) == "repeat_id"] <- "repeat"
  out
}

slow_bootstrap_values <- function(oof, W, metric, horizon) {
  values <- numeric(nrow(W))
  for (b in seq_len(nrow(W))) {
    weighted <- oof[oof$study_id %in% colnames(W), , drop = FALSE]
    weighted$weight <- as.numeric(W[b, as.character(weighted$study_id)])
    weighted <- weighted[weighted$weight > 0, , drop = FALSE]
    got <- repeat_metrics_from_oof(weighted, horizons = horizon, weight_col = "weight")
    values[b] <- mean(got[[metric]])
  }
  values
}

test_that("fast bootstrap equals slow weighted recomputation on toy OOF", {
  oof <- toy_oof()
  W <- make_bootstrap_weights(oof$study_id, B = 10, seed = 11)
  fb <- fast_bootstrap_ci(oof, metrics = c("uno_c", "auc", "brier", "harrell_c"),
                          B = 10, seed = 11, horizons = 3, include_deltas = TRUE)

  for (metric in c("uno_c", "auc", "brier", "harrell_c")) {
    slow <- slow_bootstrap_values(
      oof[oof$block == "combined", , drop = FALSE],
      W,
      metric,
      3
    )
    fast <- fb$repeat_mean_draws$value[
      fb$repeat_mean_draws$block == "combined" &
        fb$repeat_mean_draws$model == "ridge" &
        fb$repeat_mean_draws$horizon_days == 3 &
        fb$repeat_mean_draws$metric == metric
    ]
    expect_equal(fast, slow, tolerance = 1e-12)
  }
})

test_that("fast bootstrap reports nonestimable CI status conservatively", {
  oof <- toy_oof()
  oof$G_3[oof$block == "combined" & oof$duration_days > 3] <- 0.09
  fb <- fast_bootstrap_ci(oof, metrics = c("auc", "brier"), B = 5, seed = 12,
                          horizons = 3, include_deltas = FALSE)
  bad <- fb$ci[fb$ci$block == "combined", ]

  expect_true(all(bad$status == "nonestimable"))
  expect_true(all(is.na(bad$ci_low)))
  expect_true(all(is.na(bad$ci_high)))
  expect_true(all(bad$n_estimable < 5))
})

test_that("fast bootstrap point estimate equals observed OOF not bootstrap mean", {
  oof <- toy_oof()
  fb <- fast_bootstrap_ci(oof, metrics = c("brier"), B = 10, seed = 11,
                          horizons = 3, include_deltas = TRUE)
  observed <- observed_metric_points(oof, metrics = c("brier"), horizons = 3)
  row <- fb$ci[fb$ci$block == "combined" & fb$ci$model == "ridge" &
                 fb$ci$horizon_days == 3 & fb$ci$metric == "brier", ]
  boot_mean <- mean(fb$repeat_mean_draws$value[
    fb$repeat_mean_draws$block == "combined" &
      fb$repeat_mean_draws$model == "ridge" &
      fb$repeat_mean_draws$horizon_days == 3 &
      fb$repeat_mean_draws$metric == "brier"
  ])

  expect_equal(row$estimate, observed$observed[observed$block == "combined"], tolerance = 1e-12)
  expect_false(isTRUE(all.equal(row$estimate, boot_mean, tolerance = 1e-12)))
})

test_that("fast bootstrap supports IBS as a single 1825-day horizon", {
  oof <- toy_oof()
  for (i in seq_len(101)) {
    oof[[paste0("risk_grid_", i)]] <- oof$risk_3
    oof[[paste0("G_grid_", i)]] <- 1
  }
  fb <- fast_bootstrap_ci(oof, metrics = c("ibs"), B = 5, seed = 13,
                          horizons = c(1095, 1825), include_deltas = FALSE)

  expect_equal(unique(fb$ci$metric), "ibs")
  expect_equal(unique(fb$ci$horizon_days), 1825)
  expect_true(all(is.finite(fb$ci$estimate)))
})
