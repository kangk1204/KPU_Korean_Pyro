stage_genes <- c("EYA4", "ZNF568", "ZNF793", "SFMBT2", "ADHFE1",
                 "HOXA2", "BEND5", "UNC5C", "RALYL", "GFRA1")

parse_tnm_strict <- function(x) {
  x <- trimws(as.character(x))
  m <- regexec("^(T(?:[1-3]|4[ab]?))(N(?:0|1[abc]?|2[ab]?))(M0)$", x, perl = TRUE)
  got <- regmatches(x, m)[[1]]
  if (length(got) != 4L) {
    return(list(valid = FALSE, t = NA_character_, n = NA_character_, m = NA_character_,
                stage = NA_integer_, reason = "invalid_or_nonexplicit_tnm"))
  }
  t_code <- got[[2]]
  n_code <- got[[3]]
  t_num <- as.integer(substr(t_code, 2L, 2L))
  stage <- if (n_code == "N0") {
    if (t_num <= 2L) 1L else 2L
  } else {
    3L
  }
  list(valid = TRUE, t = t_code, n = n_code, m = got[[4]],
       stage = stage, reason = "valid")
}

read_stage_inputs <- function(root) {
  recurrence_path <- file.path(root, "baseline_114", "data", "ml", "recurrence.tsv")
  clinical_path <- file.path(root, "baseline_114", "data", "derived", "clinical.tsv")
  if (!file.exists(recurrence_path)) stop("missing recurrence input: ", recurrence_path)
  if (!file.exists(clinical_path)) stop("missing clinical input: ", clinical_path)
  recurrence <- read.delim(recurrence_path, check.names = FALSE, stringsAsFactors = FALSE)
  clinical <- read.delim(clinical_path, check.names = FALSE, stringsAsFactors = FALSE)
  keep <- c("study_id", "patient_id", "tnm", "stage")
  missing <- setdiff(keep, names(clinical))
  if (length(missing)) stop("clinical input missing columns: ", paste(missing, collapse = ", "))
  merge(recurrence, clinical[, keep], by = "study_id", suffixes = c("", "_clinical"),
        all.x = TRUE, sort = FALSE)
}

prepare_stage_sensitivity_data <- function(root) {
  d0 <- read_stage_inputs(root)
  primary <- d0[d0$recurrence_primary == 1L, , drop = FALSE]
  if (nrow(primary) != 82L || sum(primary$event) != 14L) {
    stop("primary recurrence cohort changed: n=", nrow(primary), " events=", sum(primary$event))
  }

  parsed <- lapply(primary$tnm, parse_tnm_strict)
  audit <- data.frame(
    study_id = primary$study_id,
    patient_id = primary$patient_id,
    event = primary$event,
    duration_days = primary$duration_days,
    provider_stage = primary$stage,
    tnm_raw = primary$tnm,
    t_code = vapply(parsed, `[[`, character(1), "t"),
    n_code = vapply(parsed, `[[`, character(1), "n"),
    m_code = vapply(parsed, `[[`, character(1), "m"),
    tnm_valid = vapply(parsed, `[[`, logical(1), "valid"),
    tnm_stage = vapply(parsed, `[[`, integer(1), "stage"),
    parse_reason = vapply(parsed, `[[`, character(1), "reason"),
    stringsAsFactors = FALSE
  )
  audit$provider_stage_iii <- as.integer(audit$provider_stage == 3L)
  audit$tnm_stage_iii <- ifelse(audit$tnm_valid, as.integer(audit$tnm_stage == 3L), NA_integer_)
  audit$provider_tnm_mismatch <- audit$tnm_valid & audit$provider_stage != audit$tnm_stage
  audit$analysis_included <- audit$tnm_valid
  audit$exclusion_reason <- ifelse(audit$tnm_valid, "", audit$parse_reason)

  valid <- audit$analysis_included
  if (sum(valid) != 79L || sum(audit$event[valid]) != 14L) {
    stop("strict TNM cohort changed: n=", sum(valid), " events=", sum(audit$event[valid]))
  }
  if (sum(audit$provider_tnm_mismatch) != 10L) {
    stop("provider/TNM mismatch count changed: ", sum(audit$provider_tnm_mismatch))
  }
  # Three records have TNM strings that cannot be parsed strictly (missing or nonstandard
  # M component); their identifiers are listed only in the restricted audit file.
  if (sum(!valid) != 3L) {
    stop("strict TNM exclusions changed: n=", sum(!valid))
  }

  stage_data <- primary[valid, , drop = FALSE]
  stage_data$provider_stage_iii <- audit$provider_stage_iii[valid]
  stage_data$tnm_stage <- audit$tnm_stage[valid]
  stage_data$tnm_stage_iii <- audit$tnm_stage_iii[valid]
  stage_data
  list(primary = primary, stage_data = stage_data, audit = audit)
}
